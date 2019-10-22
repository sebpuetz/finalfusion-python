pub mod array;

pub use array::MmapArray;

pub mod quantized;

pub use quantized::QuantizedArray;

pub mod wrappers;

pub use wrappers::StorageWrap;

use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;

use finalfusion::storage::{NdArray, Storage, StorageView};
use itertools::Itertools;
use ndarray::{Array2, Array3, Array4, ArrayBase, ArrayViewMut1, Axis, DataMut, Dimension, RemoveAxis, Array1, ArrayViewMut2, s};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::prelude::*;
use pyo3::{exceptions, PyObjectProtocol};
use rayon::ThreadPoolBuilder;

use crate::io::{find_chunk, ChunkIdentifier, Header, WriteChunk};
use crate::storage::quantized::quantize_;
use crate::util::{l2_normalize_nd, nd_to_pyobject, PyMatrix, PyQuery};
use crate::util;

/// finalfusion storage.
#[pyclass(name = Storage)]
#[derive(Clone)]
pub struct PyStorage {
    storage: Rc<StorageWrap>,
}

impl PyStorage {
    pub fn storage_(&self) -> &StorageWrap {
        self.storage.as_ref()
    }

    fn verify_idx(&self, idx: usize) -> PyResult<()> {
        if self.shape().0 <= idx {
            return Err(exceptions::IndexError::py_err(format!(
                "Index out of bounds: shape {} vs. requested {}",
                self.shape().0,
                idx
            )));
        }
        Ok(())
    }

    fn lookup_into<'a>(
        &self,
        rows: impl IntoIterator<Item = ArrayViewMut1<'a, f32>>,
        indices: impl Iterator<Item = usize>,
    ) -> PyResult<()> {
        for (idx, mut row) in indices.zip(rows.into_iter()) {
            self.verify_idx(idx)?;
            row.assign(&self.embedding(idx));
        }
        Ok(())
    }

    fn sum_into(&self, mut trg: ArrayViewMut1<f32>, indices: impl IntoIterator<Item=usize>) -> PyResult<()> {
        for idx in indices {
            self.verify_idx(idx)?;
            trg += &self.embedding(idx);
        }
        Ok(())
    }

    fn reduce_ragged_inner(&self, mut matrix: ArrayViewMut2<f32>, batch: Vec<Vec<usize>>) -> PyResult<()> {
        for (i, list) in batch.into_iter().enumerate() {
            let mut row = matrix.row_mut(i);
            let len = list.len();
            self.sum_into(row.view_mut(), list)?;
            row /= len as f32;
        }
        return Ok(())
    }

    fn batch_lookup_(&self, batch: Vec<Vec<usize>>, reduce_embeddings: bool, l2_normalize: bool) -> PyResult<PyObject> {
        let dims = self.shape().1;
        let gil = Python::acquire_gil();
        let ragged = !batch.iter().map(|l| l.len()).all_equal();
        if ragged && reduce_embeddings {
            let mut matrix = Array2::zeros([batch.len(), dims]);
            self.reduce_ragged_inner(matrix.view_mut(), batch)?;

            if l2_normalize {
                let norms = nd_to_pyobject(l2_normalize_nd(matrix.view_mut()));
                Ok((nd_to_pyobject(matrix), norms).to_object(gil.python()))
            } else {
                Ok(nd_to_pyobject(matrix))
            }
        } else {
            let batch_size = batch.len();
            let instance_size = batch[0].len();
            let embed_dims = self.shape().1;
            let mut tensor = Array3::zeros([batch_size, instance_size, embed_dims]);
            self.lookup_into(tensor.genrows_mut(), batch.into_iter().flatten())?;
            Ok(Self::maybe_reduce_and_normalize(
                tensor,
                reduce_embeddings,
                l2_normalize,
            ))
        }
    }

    fn list_lookup(&self, list: Vec<usize>, reduce_embeddings: bool, l2_normalize: bool) -> PyResult<PyObject> {
        let dims = self.shape().1;
        let gil = Python::acquire_gil();
        if reduce_embeddings {
            let mut ret = Array1::zeros(dims);
            let len = list.len();
            self.sum_into(ret.view_mut(), list)?;
            if l2_normalize {
                let norm = util::l2_normalize(ret.view_mut());
                return Ok((nd_to_pyobject(ret), norm).to_object(gil.python()));
            } else {
                ret /= len as f32;
                return Ok(nd_to_pyobject(ret))
            }
        } else {
            let mut matrix = Array2::zeros([list.len(), dims]);
            self.lookup_into(matrix.genrows_mut(), list.into_iter())?;
            if l2_normalize {
                let norms = l2_normalize_nd(matrix.view_mut()).to_pyarray(gil.python());
                return Ok((matrix.to_pyarray(gil.python()), norms).to_object(gil.python()));
            }
            Ok(Self::maybe_reduce_and_normalize(
                matrix,
                reduce_embeddings,
                l2_normalize,
            ))
        }
    }

    fn maybe_reduce_and_normalize<S, D>(
        mut tensor: ArrayBase<S, D>,
        reduce: bool,
        l2_normalize: bool,
    ) -> PyObject
    where
        S: DataMut<Elem = f32>,
        D: Dimension + RemoveAxis,
        D::Smaller: RemoveAxis,
    {
        let gil = Python::acquire_gil();
        if reduce {
            let axis = Axis(tensor.ndim() - 2);
            let mut tensor = tensor.mean_axis(axis).unwrap_or_default();
            if l2_normalize {
                let norms = l2_normalize_nd(tensor.view_mut()).to_pyarray(gil.python());
                let tensor = tensor.to_pyarray(gil.python());
                (tensor, norms).to_object(gil.python())
            } else {
                nd_to_pyobject(tensor)
            }
        } else {
            if l2_normalize {
                let norms = l2_normalize_nd(tensor.view_mut()).to_pyarray(gil.python());
                let tensor = tensor.to_pyarray(gil.python());
                (tensor, norms).to_object(gil.python())
            } else {
                nd_to_pyobject(tensor)
            }
        }
    }

    pub(crate) fn repr_(&self, start: usize) -> String {
        let padding = if start != 0 {
            start + 4 - (start + 4) % 4
        } else {
            0
        };
        let mut repr = iter::repeat(" ").take(padding - start).collect::<String>();
        let level2_padding = iter::repeat(" ").take(padding + 4).collect::<String>();
        repr += &self.chunk_identifier().to_string();
        repr += " {\n";
        repr += &level2_padding;
        repr += &format!("shape: {:?},\n", self.shape());
        let end_padding = iter::repeat(" ").take(padding).collect::<String>();
        repr += &end_padding;
        repr + "}"
    }
}

impl Deref for PyStorage {
    type Target = StorageWrap;

    fn deref(&self) -> &Self::Target {
        self.storage.as_ref()
    }
}

impl PyStorage {
    pub fn new(storage: Rc<StorageWrap>) -> Self {
        PyStorage { storage }
    }
    /// Copy storage to an array.
    ///
    /// This should only be used for storage types that do not provide
    /// an ndarray view that can be copied trivially, such as quantized
    /// storage.
    #[allow(dead_code)]
    pub(crate) fn copy_storage_to_array(storage: &dyn Storage) -> Array2<f32> {
        let (rows, dims) = storage.shape();

        let mut array = Array2::<f32>::zeros((rows, dims));
        for idx in 0..rows {
            array.row_mut(idx).assign(&storage.embedding(idx));
        }

        array
    }
}

#[pymethods]
impl PyStorage {
    #[args(mmap = "false")]
    #[new]
    fn __new__(obj: &PyRawObject, filename: &str, mmap: bool) -> PyResult<()> {
        let file = File::open(filename).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for reading: {}\n{}",
                filename, e
            ))
        })?;
        let mut read = BufReader::new(file);
        let storage_type = find_chunk(
            &mut read,
            &[ChunkIdentifier::NdArray, ChunkIdentifier::QuantizedArray],
        )?;
        match storage_type {
            ChunkIdentifier::NdArray => {
                obj.init(Self::load_array(&mut read, mmap)?);
            }
            ChunkIdentifier::QuantizedArray => {
                obj.init(Self::load_quantized(&mut read, mmap)?);
            }
            id => unreachable!("Found unreachable chunk: {}", id),
        }
        Ok(())
    }

    #[staticmethod]
    fn from_array(array: PyMatrix) -> Self {
        let array = array.0.as_array().to_owned();
        PyStorage::new(Rc::new(StorageWrap::NdArray(NdArray::new(array))))
    }

    /// quantize(self,/, n_subquantizers, n_subquantizer_bits, n_attempts, n_iterations, normalize,
    /// n_threads, verbose)
    /// --
    ///
    /// Quantize the storage.
    ///
    /// Returns a new quantized storage.
    #[allow(unused_must_use, clippy::too_many_arguments)]
    #[args(
        n_subquantizers = 0,
        n_subquantizer_bits = 8,
        n_attempts = 1,
        n_iterations = 100,
        normalize = "true",
        n_threads = 0,
        verbose = "false"
    )]
    fn quantize(
        &self,
        mut n_subquantizers: usize,
        n_subquantizer_bits: usize,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
        mut n_threads: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        if verbose {
            simple_logger::init().map_err(|_| eprintln!("Failed initializing logger."));
        }
        if n_subquantizer_bits > 8 {
            return Err(exceptions::ValueError::py_err(
                "Max value for n_subquantizer_bits is 8.",
            ));
        }
        if self.quantized() {
            return Err(exceptions::TypeError::py_err(
                "Can't quantize quantized storage.",
            ));
        }
        let (rows, cols) = self.shape();
        if n_subquantizers == 0 {
            n_subquantizers = cols / 2;
        }
        let n_subquantizer_bits = n_subquantizer_bits as u32;
        if 2usize.pow(n_subquantizer_bits) > rows {
            return Err(exceptions::RuntimeError::py_err("Cannot train with more centroids than \
            instances. Pick n_subquantizer_bits such that 2^n_subquantizer_bits <= n_instances"));
        }
        if n_threads == 0 {
            n_threads = num_cpus::get() / 2;
        }
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global();
        let quantized = match self.storage_() {
            StorageWrap::NdArray(array) => quantize_(
                array,
                n_subquantizers,
                n_subquantizer_bits,
                n_iterations,
                n_attempts,
                normalize,
            ),
            StorageWrap::MmapArray(array) => quantize_(
                array,
                n_subquantizers,
                n_subquantizer_bits,
                n_iterations,
                n_attempts,
                normalize,
            ),
            StorageWrap::QuantizedArray(_) | StorageWrap::MmapQuantizedArray(_) => unreachable!(),
        };

        Ok(PyStorage::new(Rc::new(StorageWrap::QuantizedArray(
            Box::new(quantized?),
        ))))
    }

    #[args(
        default = "PyEmbeddingDefault::default()",
        reduce_embeddings = "false",
        l2_normalize = "false"
    )]
    pub fn batch_lookup(
        &self,
        query: PyQuery<usize>,
        reduce_embeddings: bool,
        l2_normalize: bool,
    ) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let dims = self.shape().1;
        match query {
            PyQuery::Word(word_idx) => {
                self.verify_idx(word_idx)?;
                Ok(nd_to_pyobject(self.embedding(word_idx)))
            }
            PyQuery::List(list) => {
                self.list_lookup(list, reduce_embeddings, l2_normalize)
            }
            PyQuery::Batch(batch) => {
                self.batch_lookup_(batch, reduce_embeddings, l2_normalize)
            }
            PyQuery::BatchPlusOne(batch_plus_one) => {
                let outter_len = batch_plus_one.len();
                let d1_len = batch_plus_one[0].len();
                let d1_ragged = !batch_plus_one.iter().map(|batch| batch.len()).all_equal();
                let d2_len = batch_plus_one[0].first().map(|b|b.len()).unwrap_or_default();
                let d2_ragged = !batch_plus_one
                    .iter()
                    .flat_map(|batch| batch.iter())
                    .all_equal();
                if d1_ragged {
                    let mut ragged = Vec::with_capacity(outter_len);
                    for d1 in batch_plus_one {
                        ragged.push(self.batch_lookup(
                            PyQuery::Batch(d1),
                            reduce_embeddings,
                            l2_normalize,
                        )?);
                    }
                    return Ok(ragged.to_object(gil.python()))
                } else if d2_ragged {
                    let mut tensor = Array3::zeros([outter_len, d1_len, dims]);
                    for (i, batch) in batch_plus_one.into_iter().enumerate() {
                        let mut matrix = tensor.slice_mut(s!(i, .., ..));
                        self.reduce_ragged_inner(matrix.view_mut(), batch)?;
                    }
                    if l2_normalize {
                        let norms = nd_to_pyobject(l2_normalize_nd(tensor.view_mut()));
                        Ok((nd_to_pyobject(tensor), norms).to_object(gil.python()))
                    } else {
                        Ok(nd_to_pyobject(tensor))
                    }
                } else {
                    let mut tensor = Array4::zeros([outter_len, d1_len, d2_len, dims]);
                    self.lookup_into(
                        tensor.genrows_mut(),
                        batch_plus_one.into_iter().flatten().flatten(),
                    )?;
                    Ok(Self::maybe_reduce_and_normalize(
                        tensor,
                        reduce_embeddings,
                        l2_normalize,
                    ))
                }
            }
        }
    }

    /// Copy the entire embeddings matrix.
    pub fn matrix_copy(&self) -> Py<PyArray2<f32>> {
        let gil = pyo3::Python::acquire_gil();
        let matrix_view = match self.storage.as_ref() {
            StorageWrap::MmapArray(array) => array.view(),
            StorageWrap::NdArray(array) => array.view(),
            StorageWrap::QuantizedArray(quantized) => {
                let array = Self::copy_storage_to_array(quantized.as_ref());
                return array.to_pyarray(gil.python()).to_owned();
            }
            StorageWrap::MmapQuantizedArray(quantized) => {
                let array = Self::copy_storage_to_array(quantized);
                return array.to_pyarray(gil.python()).to_owned();
            }
        };

        matrix_view.to_pyarray(gil.python()).to_owned()
    }

    /// Get the shape of the storage.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.storage.shape()
    }

    /// write(self, filename,/,)
    /// --
    ///
    /// Write the storage to a file.
    pub fn write(&self, filename: &str) -> PyResult<()> {
        let file = File::create(filename).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for writing: {}\n{}",
                filename, e
            ))
        })?;
        let mut writer = BufWriter::new(file);
        Header::new(vec![self.chunk_identifier()]).write_chunk(&mut writer)?;
        self.write_chunk(&mut writer)
    }
}

#[pyproto]
impl PySequenceProtocol for PyStorage {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.shape().0)
    }

    fn __getitem__(&self, idx: isize) -> PyResult<Py<PyArray1<f32>>> {
        if idx >= self.shape().0 as isize || idx < 0 {
            Err(exceptions::IndexError::py_err("list index out of range"))
        } else {
            let gil = Python::acquire_gil();
            Ok(self
                .embedding(idx as usize)
                .into_owned()
                .to_pyarray(gil.python())
                .to_owned())
        }
    }
}

#[pyproto]
impl PyObjectProtocol for PyStorage {
    fn __repr__(&self) -> PyResult<String> {
        Ok(self.repr_(0))
    }
}

impl WriteChunk for PyStorage {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self.storage_() {
            StorageWrap::NdArray(_) => ChunkIdentifier::NdArray,
            StorageWrap::QuantizedArray(_) => ChunkIdentifier::QuantizedArray,
            StorageWrap::MmapArray(_) => ChunkIdentifier::NdArray,
            StorageWrap::MmapQuantizedArray(_) => ChunkIdentifier::QuantizedArray,
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> PyResult<()>
    where
        W: Write + Seek,
    {
        match self.storage_() {
            StorageWrap::NdArray(array) => Self::write_array_chunk(write, array.view()),
            StorageWrap::MmapArray(array) => Self::write_array_chunk(write, array.view()),
            StorageWrap::QuantizedArray(array) => {
                Self::write_quantized(write, array.quantizer(), array.embeddings(), array.norms())
            }
            StorageWrap::MmapQuantizedArray(array) => {
                Self::write_quantized(write, array.quantizer(), array.embeddings(), array.norms())
            }
        }
    }
}
