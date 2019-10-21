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

use finalfusion::prelude::{Storage, StorageView};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::prelude::*;
use pyo3::{exceptions, PyObjectProtocol};
use rayon::ThreadPoolBuilder;

use crate::io::{find_chunk, ChunkIdentifier, Header, WriteChunk};
use crate::storage::quantized::quantize_;
use finalfusion::chunks::storage::NdArray;
use crate::util::PyMatrix;

/// finalfusion storage.
#[pyclass(name=Storage)]
#[derive(Clone)]
pub struct PyStorage {
    storage: Rc<StorageWrap>,
}

impl PyStorage {
    pub fn storage_(&self) -> &StorageWrap {
        self.storage.as_ref()
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
                if mmap {
                    obj.init(Self::read_array_storage(&mut read)?);
                } else {
                    obj.init(Self::mmap_array(&mut read)?);
                }
            }
            ChunkIdentifier::QuantizedArray => {
                if mmap {
                    return Err(exceptions::NotImplementedError::py_err(
                        "Mmapped quantized matrices are not yet implemented.",
                    ));
                } else {
                    obj.init(Self::read_quantized_chunk(&mut read)?);
                }
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
            StorageWrap::QuantizedArray(_) | StorageWrap::MmapQuantizedArray => unreachable!(),
        };

        Ok(PyStorage::new(Rc::new(StorageWrap::QuantizedArray(
            Box::new(quantized?),
        ))))
    }

    /// Copy the entire embeddings matrix.
    pub fn matrix_copy(&self) -> PyResult<Py<PyArray2<f32>>> {
        let gil = pyo3::Python::acquire_gil();
        let matrix_view = match self.storage.as_ref() {
            StorageWrap::MmapArray(array) => array.view(),
            StorageWrap::NdArray(array) => array.view(),
            StorageWrap::QuantizedArray(quantized) => {
                let array = Self::copy_storage_to_array(quantized.as_ref());
                return Ok(array.to_pyarray(gil.python()).to_owned());
            }
            StorageWrap::MmapQuantizedArray => return Err(exceptions::NotImplementedError::py_err(
                "MmapQuantizedArray is not implemented yet.",
            ))
        };

        Ok(matrix_view.to_pyarray(gil.python()).to_owned())
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
            StorageWrap::MmapQuantizedArray => ChunkIdentifier::QuantizedArray,
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
            StorageWrap::MmapQuantizedArray => Err(exceptions::NotImplementedError::py_err(
                "MmapQuantizedArray is not implemented yet.",
            )),
        }
    }
}
