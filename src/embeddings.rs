use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::rc::Rc;

use finalfusion::chunks::vocab::WordIndex;
use finalfusion::compat::text::{ReadText, ReadTextDims};
use finalfusion::compat::word2vec::ReadWord2Vec;
use finalfusion::io as ffio;
use finalfusion::prelude::*;
use finalfusion::similarity::{Analogy, EmbeddingSimilarity, WordSimilarity};
use itertools::Itertools;
use ndarray::{Array1, CowArray, Ix1};
use numpy::{IntoPyArray, NpyDataType, PyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyTuple};
use pyo3::{exceptions, PyMappingProtocol};

use crate::metadata::PyMetadata;
use crate::norms::PyNorms;
use crate::similarity::similarity_results;
use crate::storage::PyStorage;
use crate::util::l2_normalize;
use crate::{PyEmbeddingIterator, PyVocab};
use finalfusion::chunks::norms::NdNorms;

/// finalfusion embeddings.
#[pyclass(name = Embeddings)]
#[derive(Clone)]
pub struct PyEmbeddings {
    // The use of Rc + RefCell should be safe in this crate:
    //
    // 1. Python is single-threaded.
    // 2. The only mutable borrow (in set_metadata) is limited
    //    to its method scope.
    // 3. None of the methods returns borrowed embeddings.
    storage: PyStorage,
    vocab: PyVocab,
    metadata: Option<PyMetadata>,
    norms: Option<PyNorms>,
}

impl PyEmbeddings {
    pub(crate) fn embedding_(&self, word: &str) -> Option<CowArray<f32, Ix1>> {
        match self.vocab.idx(word)? {
            WordIndex::Word(idx) => Some(self.storage.embedding(idx)),
            WordIndex::Subword(indices) => {
                let mut embed = Array1::zeros((self.storage.shape().1,));
                for idx in indices {
                    embed += &self.storage.embedding(idx).view();
                }

                l2_normalize(embed.view_mut());

                Some(CowArray::from(embed))
            }
        }
    }

    pub(crate) fn embedding_with_norm_(&self, word: &str) -> Option<(CowArray<f32, Ix1>, f32)> {
        match self.vocab.idx(word)? {
            WordIndex::Word(idx) => Some((
                self.storage.embedding(idx),
                self.norms().map(|n| n.0[idx]).unwrap_or(1.),
            )),
            WordIndex::Subword(indices) => {
                let mut embed = Array1::zeros((self.storage.shape().1,));
                for idx in indices {
                    embed += &self.storage.embedding(idx).view();
                }

                let norm = l2_normalize(embed.view_mut());

                Some((CowArray::from(embed), norm))
            }
        }
    }

    pub(crate) fn storage_(&self) -> &StorageWrap {
        self.storage.storage_()
    }

    pub(crate) fn vocab_(&self) -> &VocabWrap {
        self.vocab.vocab_()
    }

    pub(crate) fn metadata_(&self) -> Option<&Metadata> {
        self.metadata.as_ref().map(|metadata| metadata.metadata_())
    }

    pub(crate) fn norms_(&self) -> Option<&NdNorms> {
        self.norms.as_ref().map(|norms| norms.norms_())
    }
}

#[pymethods]
impl PyEmbeddings {
    /// Load embeddings from the given `path`.
    ///
    /// When the `mmap` argument is `True`, the embedding matrix is
    /// not loaded into memory, but memory mapped. This results in
    /// lower memory use and shorter load times, while sacrificing
    /// some query efficiency.
    #[new]
    #[args(mmap = false)]
    fn __new__(obj: &PyRawObject, path: &str, mmap: bool) -> PyResult<()> {
        // First try to load embeddings with viewable storage. If that
        // fails, attempt to load the embeddings as non-viewable
        // storage.
        let embeddings: Embeddings<VocabWrap, StorageWrap> = read_embeddings(path, mmap)
            .map_err(|err| exceptions::IOError::py_err(err.to_string()))?;

        let (metadata, vocab, storage, norms) = embeddings.into_parts();
        let vocab = PyVocab::new(Rc::new(vocab));
        let norms = norms.map(|norms| PyNorms::new(Rc::new(norms.clone())));
        let metadata = metadata.map(|metadata| PyMetadata::new(Rc::new(metadata.clone())));
        let storage = PyStorage::new(Rc::new(storage));

        obj.init(PyEmbeddings {
            storage,
            vocab,
            metadata,
            norms,
        });

        Ok(())
    }

    /// read_fasttext(path,/ lossy)
    /// --
    ///
    /// Read embeddings in the fasttext format.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_fasttext(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_fasttext_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_fasttext(r))
        }
    }

    /// read_text(path,/ lossy)
    /// --
    ///
    /// Read embeddings in text format. This format uses one line per
    /// embedding. Each line starts with the word in UTF-8, followed
    /// by its vector components encoded in ASCII. The word and its
    /// components are separated by spaces.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_text(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text(r))
        }
    }

    /// read_text_dims(path,/ lossy)
    /// --
    ///
    /// Read embeddings in text format with dimensions. In this format,
    /// the first line states the shape of the embedding matrix. The
    /// number of rows (words) and columns (embedding dimensionality) is
    /// separated by a space character. The remainder of the file uses
    /// one line per embedding. Each line starts with the word in UTF-8,
    /// followed by its vector components encoded in ASCII. The word and
    /// its components are separated by spaces.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_text_dims(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text_dims_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text_dims(r))
        }
    }

    /// read_word2vec(path,/ lossy)
    /// --
    ///
    /// Read embeddings in the word2vec binary format.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_word2vec(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_word2vec_binary_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_word2vec_binary(r))
        }
    }

    /// Embeddings metadata.
    fn metadata(&self) -> Option<PyMetadata> {
        self.metadata.clone()
    }

    /// Get the model's norms.
    fn norms(&self) -> Option<PyNorms> {
        self.norms.clone()
    }

    /// Get the model's vocabulary.
    fn vocab(&self) -> PyVocab {
        self.vocab.clone()
    }

    /// Get the model's storage.
    fn storage(&self) -> PyStorage {
        self.storage.clone()
    }

    /// Perform an anology query.
    ///
    /// This returns words for the analogy query *w1* is to *w2*
    /// as *w3* is to ?.
    #[args(limit = 10, mask = "(true, true, true)")]
    fn analogy(
        &self,
        py: Python,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        mask: (bool, bool, bool),
    ) -> PyResult<Vec<PyObject>> {
        use StorageWrap::*;
        match self.storage.storage_() {
            MmapQuantizedArray(_) | QuantizedArray(_) => {
                return Err(exceptions::ValueError::py_err(
                    "Analogy queries are not supported for this type of embedding matrix",
                ))
            }
            _ => (),
        };

        let results = self
            .analogy_masked([word1, word2, word3], [mask.0, mask.1, mask.2], limit)
            .map_err(|lookup| {
                let failed = [word1, word2, word3]
                    .iter()
                    .zip(lookup.iter())
                    .filter(|(_, success)| !*success)
                    .map(|(word, _)| word)
                    .join(" ");
                exceptions::KeyError::py_err(format!("Unknown word or n-grams: {}", failed))
            })?;

        similarity_results(py, results)
    }

    /// embedding(word,/, default)
    /// --
    ///
    /// Get the embedding for the given word.
    ///
    /// If the word is not known, its representation is approximated
    /// using subword units. #
    ///
    /// If no representation can be calculated:
    ///  - `None` if `default` is `None`
    ///  - an array filled with `default` if `default` is a scalar
    ///  - an array if `default` is a 1-d array
    ///  - an array filled with values from `default` if it is an iterator over floats.
    #[args(default = "PyEmbeddingDefault::default()")]
    fn embedding(
        &self,
        word: &str,
        default: PyEmbeddingDefault,
    ) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let gil = pyo3::Python::acquire_gil();
        if let PyEmbeddingDefault::Embedding(array) = &default {
            if array.as_ref(gil.python()).shape()[0] != self.storage.shape().1 {
                return Err(exceptions::ValueError::py_err(format!(
                    "Invalid shape of default embedding: {}",
                    array.as_ref(gil.python()).shape()[0]
                )));
            }
        }

        if let Some(embedding) = self.embedding_(word) {
            let embedding = embedding.to_owned();

            return Ok(Some(
                PyArray1::from_owned_array(gil.python(), embedding).to_owned(),
            ));
        };
        match default {
            PyEmbeddingDefault::Constant(constant) => {
                let nd_arr = Array1::from_elem([self.storage.shape().1], constant);
                Ok(Some(nd_arr.into_pyarray(gil.python()).to_owned()))
            }
            PyEmbeddingDefault::Embedding(array) => Ok(Some(array)),
            PyEmbeddingDefault::None => Ok(None),
        }
    }

    fn embedding_with_norm(&self, word: &str) -> Option<Py<PyTuple>> {
        let embedding_with_norm = self.embedding_with_norm_(word);

        embedding_with_norm.map(|(embedding, norm)| {
            let gil = pyo3::Python::acquire_gil();
            let embedding = embedding.into_owned().into_pyarray(gil.python());
            (embedding, norm).into_py(gil.python())
        })
    }

    /// Perform a similarity query.
    #[args(limit = 10)]
    fn word_similarity(&self, py: Python, word: &str, limit: usize) -> PyResult<Vec<PyObject>> {
        use StorageWrap::*;
        match self.storage.storage_() {
            MmapQuantizedArray(_) | QuantizedArray(_) => {
                return Err(exceptions::ValueError::py_err(
                    "Similarity queries are not supported for this type of embedding matrix",
                ))
            }
            _ => (),
        };
        let results = <Self as WordSimilarity>::word_similarity(&self, word, limit)
            .ok_or_else(|| exceptions::KeyError::py_err("Unknown word and n-grams"))?;

        similarity_results(py, results)
    }

    /// Perform a similarity query based on a query embedding.
    #[args(limit = 10, skip = "Skips(HashSet::new())")]
    fn embedding_similarity(
        &self,
        py: Python,
        embedding: PyEmbedding,
        skip: Skips,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        use StorageWrap::*;
        match self.storage.storage_() {
            MmapQuantizedArray(_) | QuantizedArray(_) => {
                return Err(exceptions::ValueError::py_err(
                    "Similarity queries are not supported for this type of embedding matrix",
                ))
            }
            _ => (),
        };

        let embedding = embedding.0.as_array();
        if embedding.shape()[0] != self.storage_().shape().1 {
            return Err(exceptions::ValueError::py_err(format!(
                "Incompatible embedding shapes: embeddings: ({},), query: ({},)",
                embedding.shape()[0],
                self.storage_().shape().1
            )));
        }

        let results = self.embedding_similarity_masked(embedding, limit, &skip.0);

        similarity_results(
            py,
            results.ok_or_else(|| exceptions::KeyError::py_err("Unknown word and n-grams"))?,
        )
    }

    /// Write the embeddings to a finalfusion file.
    fn write(&self, filename: &str) -> PyResult<()> {
        let f = File::create(filename)?;
        let mut writer = BufWriter::new(f);
        let vocab = self.vocab_().clone();
        let storage = match self.storage_() {
            StorageWrap::QuantizedArray(quant) => PyStorage::copy_storage_to_array(quant.as_ref()),
            StorageWrap::MmapQuantizedArray(quant) => PyStorage::copy_storage_to_array(quant),
            StorageWrap::NdArray(array) => array.view().to_owned(),
            StorageWrap::MmapArray(array) => array.view().to_owned(),
        };
        let metadata = self.metadata_().cloned();
        let norms = self.norms_().cloned().unwrap_or_else(|| {
            let norms = Array1::ones([self.vocab_().words_len()]);
            NdNorms(norms)
        });

        Embeddings::new(metadata, vocab, NdArray::new(storage), norms)
            .write_embeddings(&mut writer)
            .map_err(|err| exceptions::IOError::py_err(err.to_string()))
    }
}

#[pyproto]
impl PyMappingProtocol for PyEmbeddings {
    fn __getitem__(&self, word: &str) -> PyResult<Py<PyArray1<f32>>> {
        match self.embedding_(word) {
            Some(embedding) => {
                let gil = pyo3::Python::acquire_gil();
                Ok(embedding.into_owned().into_pyarray(gil.python()).to_owned())
            }
            None => Err(exceptions::KeyError::py_err("Unknown word and n-grams")),
        }
    }
}

#[pyproto]
impl PyIterProtocol for PyEmbeddings {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let iter = IntoPy::into_py(Py::new(py, PyEmbeddingIterator::new(slf.clone(), 0))?, py);

        Ok(iter)
    }
}

fn read_embeddings<S>(path: &str, mmap: bool) -> Result<Embeddings<VocabWrap, S>, ffio::Error>
where
    Embeddings<VocabWrap, S>: ReadEmbeddings + MmapEmbeddings,
{
    let f = File::open(path)
        .map_err(|e| ffio::ErrorKind::io_error("Cannot open embeddings file for reading", e))?;
    let mut reader = BufReader::new(f);

    let embeddings = if mmap {
        Embeddings::mmap_embeddings(&mut reader)?
    } else {
        Embeddings::read_embeddings(&mut reader)?
    };

    Ok(embeddings)
}

fn read_non_fifu_embeddings<R, V>(path: &str, read_embeddings: R) -> PyResult<PyEmbeddings>
where
    R: FnOnce(&mut BufReader<File>) -> ffio::Result<Embeddings<V, NdArray>>,
    V: Vocab + Clone + Into<VocabWrap>,
    Embeddings<VocabWrap, StorageViewWrap>: From<Embeddings<V, NdArray>>,
{
    let f = File::open(path).map_err(|err| {
        exceptions::IOError::py_err(format!(
            "Cannot read text embeddings from '{}': {}'",
            path, err
        ))
    })?;
    let mut reader = BufReader::new(f);

    let embeddings = read_embeddings(&mut reader).map_err(|err| {
        exceptions::IOError::py_err(format!(
            "Cannot read text embeddings from '{}': {}'",
            path, err
        ))
    })?;
    let (metadata, vocab, storage, norms) = embeddings.into_parts();
    let storage = PyStorage::new(Rc::new(storage.into()));
    let vocab = PyVocab::new(Rc::new(vocab.into()));
    let norms = norms.map(|norms| PyNorms::new(Rc::new(norms)));
    let metadata = metadata.map(|metadata| PyMetadata::new(Rc::new(metadata)));

    Ok(PyEmbeddings {
        storage,
        vocab,
        norms,
        metadata,
    })
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

fn collect_array_from_py_iter(iter: PyIterator, len: Option<usize>) -> PyResult<Py<PyArray1<f32>>> {
    let mut embed_vec = len.map(Vec::with_capacity).unwrap_or_default();
    for item in iter {
        let item = item.and_then(|item| item.extract())?;
        embed_vec.push(item);
    }
    let gil = Python::acquire_gil();
    let embed = PyArray1::from_vec(gil.python(), embed_vec).to_owned();
    Ok(embed)
}

struct Skips<'a>(HashSet<&'a str>);

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

struct PyEmbedding<'a>(&'a PyArray1<f32>);

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
