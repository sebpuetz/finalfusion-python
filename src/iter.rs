use finalfusion::prelude::*;
use numpy::{IntoPyArray, PyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;

use crate::embeddings::PyEmbeddings;

#[pyclass(name=EmbeddingIterator)]
pub struct PyEmbeddingIterator {
    embeddings: PyEmbeddings,
    idx: usize,
}

impl PyEmbeddingIterator {
    pub fn new(embeddings: PyEmbeddings, idx: usize) -> Self {
        PyEmbeddingIterator { embeddings, idx }
    }
}

#[pyproto]
impl PyIterProtocol for PyEmbeddingIterator {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<Py<PyEmbeddingIterator>> {
        Ok(slf.into())
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<PyEmbedding>> {
        let slf = &mut *slf;

        let embeddings = &slf.embeddings;
        let vocab = embeddings.vocab_();

        if slf.idx < vocab.words_len() {
            let word = vocab.words()[slf.idx].to_string();
            let embed = embeddings.storage_().embedding(slf.idx);
            let norm = embeddings.norms_().map(|n| n.0[slf.idx]).unwrap_or(1.);

            slf.idx += 1;

            let gil = pyo3::Python::acquire_gil();
            Ok(Some(PyEmbedding {
                word,
                embedding: embed.into_owned().into_pyarray(gil.python()).to_owned(),
                norm,
            }))
        } else {
            Ok(None)
        }
    }
}

/// A word and its embedding and embedding norm.
#[pyclass(name=Embedding)]
pub struct PyEmbedding {
    embedding: Py<PyArray1<f32>>,
    norm: f32,
    word: String,
}

#[pymethods]
impl PyEmbedding {
    /// Get the embedding.
    #[getter]
    pub fn get_embedding(&self) -> Py<PyArray1<f32>> {
        let gil = Python::acquire_gil();
        self.embedding.clone_ref(gil.python())
    }

    /// Get the word.
    #[getter]
    pub fn get_word(&self) -> &str {
        &self.word
    }

    /// Get the norm.
    #[getter]
    pub fn get_norm(&self) -> f32 {
        self.norm
    }
}
