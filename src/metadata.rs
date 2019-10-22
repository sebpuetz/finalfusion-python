use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use finalfusion::metadata::Metadata;
use pyo3::prelude::*;
use pyo3::{exceptions, PyObjectProtocol, PyResult};
use toml::Value;

use crate::io::{find_chunk, ChunkIdentifier, ReadChunk, WriteChunk};
use pyo3::types::PyAny;

/// finalfusion storage.
#[pyclass(name=Metadata)]
#[derive(Clone)]
pub struct PyMetadata {
    metadata: Rc<Metadata>,
}

impl PyMetadata {
    pub fn new(metadata: Rc<Metadata>) -> Self {
        PyMetadata { metadata }
    }

    pub(crate) fn repr_(&self, start: usize) -> String {
        let padding = if start != 0 {
            start + 4 - (start + 4) % 4
        } else {
            0
        };
        let repr = iter::repeat(" ").take(padding - start).collect::<String>();
        repr + &format!("{:?}", &self.to_string())
    }
}

impl Deref for PyMetadata {
    type Target = Metadata;

    fn deref(&self) -> &Self::Target {
        self.metadata.as_ref()
    }
}

#[pymethods]
impl PyMetadata {
    #[new]
    fn __new__(obj: &PyRawObject, metadata: &str) -> PyResult<()> {
        let value = match metadata.parse::<Value>() {
            Ok(value) => value,
            Err(err) => {
                return Err(exceptions::ValueError::py_err(format!(
                    "Metadata is invalid TOML: {}",
                    err
                )));
            }
        };
        obj.init(PyMetadata {
            metadata: Rc::new(value.into()),
        });
        Ok(())
    }

    /// from_embeddings(path,/,)
    /// --
    ///
    /// Read metadata from an embeddings file.
    #[staticmethod]
    fn from_embeddings(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for reading: {}\n{}",
                path, e
            ))
        })?;
        let mut read = BufReader::new(file);
        find_chunk(&mut read, &[ChunkIdentifier::Metadata])?;
        Self::read_chunk(&mut read)
    }
}

#[pyproto]
impl<'a> PyObjectProtocol<'a> for PyMetadata {
    fn __str__(&self) -> PyResult<String> {
        Ok(toml::ser::to_string_pretty(self.metadata.as_ref().deref())
            .map_err(|e| exceptions::IOError::py_err(format!("Metadata is invalid TOML: {}", e)))?)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.repr_(0))
    }
}

impl ReadChunk for PyMetadata {
    fn read_chunk<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::Metadata)?;

        let chunk_len = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read metadata chunk length\n{}", e))
        })? as usize;
        let mut buf = vec![0; chunk_len];
        read.read_exact(&mut buf)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot read metadata\n{}", e)))?;
        let buf_str = String::from_utf8(buf).map_err(|e| {
            exceptions::ValueError::py_err(format!("TOML metadata contains invalid UTF-8\n{}", e))
        })?;
        let metadata = buf_str.parse::<Value>().map_err(|e| {
            exceptions::ValueError::py_err(format!("Cannot deserialize TOML metadata: {}", e))
        })?;
        Ok(PyMetadata::new(Rc::new(metadata.into())))
    }
}

impl WriteChunk for PyMetadata {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::Metadata
    }

    fn write_chunk<W>(&self, write: &mut W) -> PyResult<()>
    where
        W: Write,
    {
        let metadata_str = self.to_string();
        write
            .write_u32::<LittleEndian>(self.chunk_identifier() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write metadata chunk identifier\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(metadata_str.len() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write metadata chunk length\n{}", e))
            })?;
        write
            .write_all(metadata_str.as_bytes())
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write metadata\n{}", e)))?;
        Ok(())
    }
}

impl<'a> FromPyObject<'a> for PyMetadata {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let metadata = ob.downcast_ref::<PyMetadata>()?;
        Ok(metadata.clone())
    }
}