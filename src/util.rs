use ndarray::ArrayViewMut1;
use numpy::{NpyDataType, PyArray1, PyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator};
use std::collections::HashSet;

pub fn l2_normalize(mut v: ArrayViewMut1<f32>) -> f32 {
    let norm = v.dot(&v).sqrt();

    if norm != 0. {
        v /= norm;
    }

    norm
}

pub enum PyEmbeddingDefault {
    Embedding(Py<PyArray1<f32>>),
    Constant(f32),
    None,
}

impl<'a> Default for PyEmbeddingDefault {
    fn default() -> Self {
        PyEmbeddingDefault::None
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
            return Ok(PyMatrix(matrix))
        }
        if let Ok(matrix) = ob.extract::<&PyArray2<f64>>() {
            return Ok(PyMatrix(matrix.cast::<f32>(false)?))
        }
        return Err(exceptions::TypeError::py_err(format!("Expected 2-d float array.",)));
    }
}
