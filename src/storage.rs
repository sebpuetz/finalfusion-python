use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use finalfusion::chunks::storage::NdArray;
use finalfusion::prelude::{Storage, StorageView, StorageWrap};
use ndarray::{Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;

use crate::io::{find_chunk, padding, ChunkIdentifier, Header, TypeId, WriteChunk};
use std::fs::File;
use std::mem::size_of;

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
    #[new]
    fn __new__(obj: &PyRawObject, filename: &str) -> PyResult<()> {
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
                obj.init(Self::read_array_storage(&mut read)?);
                Ok(())
            }
            ChunkIdentifier::QuantizedArray => unimplemented!(),
            id => unreachable!("Found unreachable chunk: {}", id),
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

impl PyStorage {
    fn read_array_storage<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read embedding matrix chunk length\n{}", e))
        })?;

        let rows = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of rows of the embedding matrix\n{}",
                e
            ))
        })? as usize;
        let cols = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of columns of the embedding matrix\n{}",
                e
            ))
        })? as usize;

        // The components of the embedding matrix should be of type f32.
        f32::ensure_data_type(read)?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip padding\n{}", e)))?;

        let mut data = vec![0f32; rows * cols];
        read.read_f32_into::<LittleEndian>(&mut data).map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read embedding matrix\n{}", e))
        })?;

        let array = Array2::from_shape_vec((rows, cols), data).map_err(|e| {
            exceptions::ValueError::py_err(format!("Invalid storage shape:\n{}", e))
        })?;
        Ok(PyStorage::new(Rc::new(NdArray::new(array).into())))
    }

    fn write_array_chunk<W>(write: &mut W, data: ArrayView2<f32>) -> PyResult<()>
    where
        W: Seek + Write,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write embedding matrix chunk identifier\n{}",
                    e
                ))
            })?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);

        // Chunk size: rows (u64), columns (u32), type id (u32),
        //             padding ([0,4) bytes), matrix.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + n_padding as usize
            + (data.nrows() * data.ncols() * size_of::<f32>());
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write embedding matrix chunk length\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(data.nrows() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write number of rows of the embedding matrix\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(data.ncols() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write number of columns of the embedding matrix\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write embeddings matrix type identifier\n{}",
                    e
                ))
            })?;

        // Write padding, such that the embedding matrix starts on at
        // a multiple of the size of f32 (4 bytes). This is necessary
        // for memory mapping a matrix. Interpreting the raw u8 data
        // as a proper f32 array requires that the data is aligned in
        // memory. However, we cannot always memory map the starting
        // offset of the matrix directly, since mmap(2) requires a
        // file offset that is page-aligned. Since the page size is
        // always a larger power of 2 (e.g. 2^12), which is divisible
        // by 4, the offset of the matrix with regards to the page
        // boundary is also a multiple of 4.

        let padding = vec![0; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write padding\n{}", e)))?;

        for row in data.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col).map_err(|e| {
                    exceptions::IOError::py_err(format!(
                        "Cannot write embedding matrix component\n{}",
                        e
                    ))
                })?;
            }
        }

        Ok(())
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
            StorageWrap::QuantizedArray(_) => unimplemented!(),
            StorageWrap::MmapQuantizedArray(_) => unimplemented!(),
        }
    }
}
