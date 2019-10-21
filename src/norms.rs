use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::ops::Deref;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use finalfusion::chunks::norms::NdNorms;
use itertools::Itertools;
use ndarray::Array1;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::{exceptions, PyObjectProtocol, PyResult, PySequenceProtocol};

use crate::io::{find_chunk, padding, ChunkIdentifier, Header, ReadChunk, TypeId, WriteChunk};
use std::iter;

/// finalfusion storage.
#[pyclass(name = Norms)]
#[derive(Clone)]
pub struct PyNorms {
    norms: Rc<NdNorms>,
}

impl PyNorms {
    pub fn new(norms: Rc<NdNorms>) -> Self {
        PyNorms { norms }
    }

    pub(crate) fn norms_(&self) -> &NdNorms {
        self.norms.as_ref()
    }

    pub(crate) fn repr_(&self, start: usize) -> String {
        let padding = if start != 0 {
            start + 4 - (start + 4) % 4
        } else {
            0
        };
        let mut repr = iter::repeat(" ").take(padding - start).collect::<String>();
        let level2_padding = iter::repeat(" ").take(padding + 4).collect::<String>();
        repr += "Norms {\n";
        repr += &level2_padding;
        repr += &format!("len: {},\n", self.len());
        repr += &level2_padding;
        if self.len() > 10 {
            repr += &format!(
                "norms: [{},...],\n",
                self.iter().take(10).map(|&n| n.to_string()).join(", ")
            );
        } else {
            repr += &format!(
                "norms: [{}],\n",
                self.iter().map(|&n| n.to_string()).join(", ")
            );
        }
        repr += &iter::repeat(" ").take(padding).collect::<String>();;
        repr.push('}');
        repr
    }
}

impl Deref for PyNorms {
    type Target = NdNorms;

    fn deref(&self) -> &Self::Target {
        self.norms.as_ref()
    }
}

#[pymethods]
impl PyNorms {
    #[allow(unused_must_use)]
    #[new]
    fn __new__(obj: &PyRawObject, filename: &str) -> PyResult<()> {
        let file = File::open(filename).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for reading: {}\n{}",
                filename, e
            ))
        })?;
        let mut read = BufReader::new(file);
        find_chunk(&mut read, &[ChunkIdentifier::NdNorms]);
        obj.init(Self::read_chunk(&mut read)?);
        Ok(())
    }

    /// from_array(array,/,)
    /// --
    ///
    /// Create new Norms from an array or list.
    #[staticmethod]
    fn from_array(obj: &PyAny) -> PyResult<Self> {
        match obj.extract::<&PyArray1<f32>>() {
            Ok(n) => return Ok(PyNorms::new(Rc::new(n.as_array().to_owned().into()))),
            Err(_) => {
                if let Ok(n) = obj.extract().map(|n: &PyArray1<f64>| {
                    let norms = n
                        .as_array()
                        .iter()
                        .map(|&norm| norm as f32)
                        .collect::<Vec<_>>();
                    PyNorms::new(Rc::new(Array1::from(norms).into()))
                }) {
                    return Ok(n);
                }
            }
        }

        if let Ok(list) = obj.extract::<&PyList>() {
            let mut norms = Vec::with_capacity(list.len());
            for item in list {
                norms.push(item.extract()?);
            }
            return Ok(PyNorms::new(Rc::new(norms.into())));
        }
        Err(exceptions::TypeError::py_err("failed to construct norms."))
    }

    /// write(self, filename,/,)
    /// --
    ///
    /// Write norms to the given file.
    fn write(&self, filename: &str) -> PyResult<()> {
        let file = File::create(filename).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for writing: {}\n{}",
                filename, e
            ))
        })?;
        let mut writer = BufWriter::new(file);
        Header::new(vec![ChunkIdentifier::NdNorms]).write_chunk(&mut writer)?;
        self.write_chunk(&mut writer)
    }

    /// to_numpy(self,/,)
    /// --
    ///
    /// Copy the backing array to a numpy array.
    fn to_numpy(&self) -> Py<PyArray1<f32>> {
        let gil = Python::acquire_gil();
        self.view().to_pyarray(gil.python()).to_owned()
    }
}

#[pyproto]
impl<'a> PySequenceProtocol<'a> for PyNorms {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn __getitem__(&self, idx: isize) -> PyResult<f32> {
        if idx >= self.len() as isize || idx < 0 {
            Err(exceptions::IndexError::py_err("list index out of range"))
        } else {
            Ok(self[idx as usize])
        }
    }
}

#[pyproto]
impl<'a> PyObjectProtocol<'a> for PyNorms {
    fn __repr__(&self) -> PyResult<String> {
        Ok(self.repr_(0))
    }
}

impl WriteChunk for PyNorms {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdNorms
    }

    fn write_chunk<W>(&self, write: &mut W) -> PyResult<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdNorms as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write norms chunk identifier\n{}", e))
            })?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);

        // Chunk size: len (u64), type id (u32), padding ([0,4) bytes), vector.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + n_padding as usize
            + (self.len() * size_of::<f32>());
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write norms chunk length\n{}", e))
            })?;
        write
            .write_u64::<LittleEndian>(self.len() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write norms vector length\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write norms vector type identifier\n{}",
                    e
                ))
            })?;

        let padding = vec![0; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write padding\n{}", e)))?;

        for &val in self.iter() {
            write
                .write_f32::<LittleEndian>(val)
                .map_err(|e| exceptions::IOError::py_err(format!("Cannot write norm\n{}", e)))?;
        }

        Ok(())
    }
}

impl ReadChunk for PyNorms {
    fn read_chunk<R>(read: &mut R) -> Result<Self, PyErr>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdNorms)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read norms chunk length\n{}", e))
        })?;

        let len = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read norms vector length\n{}", e))
        })? as usize;

        f32::ensure_data_type(read).map_err(|e| exceptions::IOError::py_err(e.to_string()))?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);

        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip padding\n{}", e)))?;

        let mut data = vec![0f32; len];
        read.read_f32_into::<LittleEndian>(&mut data)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot read norms\n{}", e)))?;
        Ok(PyNorms::new(Rc::new(data.into())))
    }
}
