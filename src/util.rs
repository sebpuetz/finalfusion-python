use crate::io::padding;
use memmap::{Mmap, MmapOptions};
use ndarray::{ArrayViewMut1, ArrayViewMut, Dimension, Array, ArrayBase, Data, Axis, RemoveAxis};
use numpy::{NpyDataType, PyArray1, PyArray2, ToPyArray, TypeNum};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};

pub fn nd_to_pyobject<S, D, A>(a: ArrayBase<S, D>) -> PyObject
    where
        S: Data<Elem = A>,
        D: Dimension,
        A: TypeNum,
{
    let gil = Python::acquire_gil();
    a.to_pyarray(gil.python()).to_object(gil.python())
}

pub fn l2_normalize(mut v: ArrayViewMut1<f32>) -> f32 {
    let norm = v.dot(&v).sqrt();

    if norm != 0. {
        v /= norm;
    }

    norm
}

pub fn l2_normalize_nd<D>(mut v: ArrayViewMut<f32, D>) -> Array<f32, D::Smaller>
    where D: Dimension + RemoveAxis {
    let mut norms = Vec::with_capacity(v.shape().len());
    for row in v.genrows_mut() {
        norms.push(l2_normalize(row))
    }

    let norms_shape = v.raw_dim().remove_axis(Axis(v.ndim()-1));
    Array::from_shape_vec(norms_shape, norms)
        .expect("Invalid shapes after normalization")
}

pub fn mmap_array(read: &mut BufReader<File>, array_len: usize) -> PyResult<Mmap> {
    let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
        exceptions::IOError::py_err(format!(
            "Cannot get file position for computing padding\n{}",
            e
        ))
    })?);
    read.seek(SeekFrom::Current(n_padding as i64))
        .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip padding\n{}", e)))?;

    // Set up memory mapping.
    let offset = read.seek(SeekFrom::Current(0)).map_err(|e| {
        exceptions::IOError::py_err(format!(
            "Cannot get file position for memory mapping embedding matrix\n{}",
            e,
        ))
    })?;
    let mut mmap_opts = MmapOptions::new();
    let map = unsafe {
        mmap_opts
            .offset(offset)
            .len(array_len)
            .map(&read.get_ref())
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot memory map array\n{}", e)))?
    };
    // Position the reader after the matrix.
    read.seek(SeekFrom::Current(array_len as i64))
        .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip mmapped array\n{}", e)))?;
    Ok(map)
}

pub enum PyEmbeddingDefault {
    Embedding(Py<PyArray1<f32>>),
    Constant(f32),
    None,
}

impl Default for PyEmbeddingDefault {
    fn default() -> Self {
        PyEmbeddingDefault::None
    }
}

impl Clone for PyEmbeddingDefault {
    fn clone(&self) -> Self {
        match self {
            PyEmbeddingDefault::Constant(c) => PyEmbeddingDefault::Constant(*c),
            PyEmbeddingDefault::Embedding(e) => {
                let gil = Python::acquire_gil();
                PyEmbeddingDefault::Embedding(
                    e.as_ref(gil.python())
                        .as_array()
                        .to_pyarray(gil.python())
                        .to_owned(),
                )
            }
            PyEmbeddingDefault::None => PyEmbeddingDefault::None,
        }
    }
}

impl<'a> FromPyObject<'a> for PyEmbeddingDefault {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        if ob.is_none() {
            return Ok(PyEmbeddingDefault::None);
        }
        if let Ok(emb) = ob
            .extract()
            .map(|e: &PyArray1<f32>| PyEmbeddingDefault::Embedding(e.to_owned()))
        {
            return Ok(emb);
        }

        if let Ok(constant) = ob.extract().map(PyEmbeddingDefault::Constant) {
            return Ok(constant);
        }
        if let Ok(embed) = ob
            .iter()
            .and_then(|iter| collect_array_from_py_iter(iter, ob.len().ok()))
            .map(PyEmbeddingDefault::Embedding)
        {
            return Ok(embed);
        }

        Err(exceptions::TypeError::py_err(
            "failed to construct default value.",
        ))
    }
}

pub(crate) fn collect_array_from_py_iter(
    iter: PyIterator,
    len: Option<usize>,
) -> PyResult<Py<PyArray1<f32>>> {
    let mut embed_vec = len.map(Vec::with_capacity).unwrap_or_default();
    for item in iter {
        let item = item.and_then(|item| item.extract())?;
        embed_vec.push(item);
    }
    let gil = Python::acquire_gil();
    let embed = PyArray1::from_vec(gil.python(), embed_vec).to_owned();
    Ok(embed)
}

pub(crate) struct Skips<'a>(pub HashSet<&'a str>);

impl<'a> FromPyObject<'a> for Skips<'a> {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let mut set = ob.len().map(HashSet::with_capacity).unwrap_or_default();
        if ob.is_none() {
            return Ok(Skips(set));
        }
        for el in ob
            .iter()
            .map_err(|_| exceptions::TypeError::py_err("Iterable expected"))?
        {
            let el = el?;
            set.insert(el.extract().map_err(|_| {
                exceptions::TypeError::py_err(format!("Expected String not: {}", el))
            })?);
        }
        Ok(Skips(set))
    }
}

pub(crate) struct PyEmbedding<'a>(pub &'a PyArray1<f32>);

impl<'a> FromPyObject<'a> for PyEmbedding<'a> {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let embedding = ob
            .downcast_ref::<PyArray1<f32>>()
            .map_err(|_| exceptions::TypeError::py_err("Expected array with dtype Float32"))?;
        if embedding.data_type() != NpyDataType::Float32 {
            return Err(exceptions::TypeError::py_err(format!(
                "Expected dtype Float32, got {:?}",
                embedding.data_type()
            )));
        };
        Ok(PyEmbedding(embedding))
    }
}

pub(crate) struct PyMatrix<'a>(pub &'a PyArray2<f32>);

impl<'a> FromPyObject<'a> for PyMatrix<'a> {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        if let Ok(matrix) = ob.extract::<&PyArray2<f32>>() {
            return Ok(PyMatrix(matrix));
        }
        if let Ok(matrix) = ob.extract::<&PyArray2<f64>>() {
            return Ok(PyMatrix(matrix.cast::<f32>(false)?));
        }
        Err(exceptions::TypeError::py_err("Expected 2-d float array."))
    }
}

#[derive(Debug)]
pub enum PyQuery<V> {
    Word(V),
    List(Vec<V>),
    Batch(Vec<Vec<V>>),
    BatchPlusOne(Vec<Vec<Vec<V>>>),
}

impl<'a, V> FromPyObject<'a> for PyQuery<V>
where
    V: FromPyObject<'a>,
{
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(string) = ob.extract() {
            return Ok(PyQuery::Word(string));
        }

        if let Ok(first) = ob.extract() {
            return Ok(PyQuery::List(first));
        }
        if let Ok(second) = ob.extract() {
            return Ok(PyQuery::Batch(second));
        }
        if let Ok(third) = ob.extract() {
            return Ok(PyQuery::BatchPlusOne(third));
        }
        Err(exceptions::ValueError::py_err(
            "Expected scalar, list of scalars of nested list of scalars.",
        ))
    }
}
