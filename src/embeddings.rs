use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::mem;
use std::rc::Rc;

use finalfusion::norms::NdNorms;
use finalfusion::storage::{NdArray, Storage};
use finalfusion::vocab::{Vocab, WordIndex};
use finalfusion::io as ffio;
use finalfusion::prelude::{
    Embeddings, ReadFastText, ReadText, ReadTextDims, ReadWord2Vec,
    VocabWrap,
};
use finalfusion::similarity::{Analogy, EmbeddingSimilarity, WordSimilarity};
use itertools::Itertools;
use ndarray::{Array1, CowArray, Ix1};
use numpy::{IntoPyArray, PyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{exceptions, PyMappingProtocol, PyObjectProtocol};

use crate::io::{ChunkIdentifier, Header, ReadChunk, WriteChunk};
use crate::metadata::PyMetadata;
use crate::norms::PyNorms;
use crate::similarity::similarity_results;
use crate::storage::{PyStorage, StorageWrap};
use crate::util::{l2_normalize, PyEmbedding, PyEmbeddingDefault, Skips, PyQuery};
use crate::{io, PyEmbeddingIterator, PyVocab};

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
    storage: Option<PyStorage>,
    vocab: Option<PyVocab>,
    metadata: Option<PyMetadata>,
    norms: Option<PyNorms>,
}

impl PyEmbeddings {
    fn empty() -> Self {
        PyEmbeddings {
            storage: None,
            vocab: None,
            metadata: None,
            norms: None,
        }
    }

    pub(crate) fn embedding_(&self, word: &str) -> Option<CowArray<f32, Ix1>> {
        let vocab = self.vocab_()?;
        let storage = self.storage_()?;
        vocab.idx(word).map(|idx| match idx {
            WordIndex::Word(idx) => storage.embedding(idx),
            WordIndex::Subword(indices) => {
                let mut embed = Array1::zeros((storage.shape().1,));
                for idx in indices {
                    embed += &storage.embedding(idx).view();
                }

                l2_normalize(embed.view_mut());

                CowArray::from(embed)
            }
        })
    }

    pub(crate) fn embedding_with_norm_(&self, word: &str) -> Option<(CowArray<f32, Ix1>, f32)> {
        let storage = self.storage_()?;
        let vocab = self.vocab_()?;
        let norms = self.norms_()?;
        vocab.idx(word).map(|idx| match idx {
            WordIndex::Word(idx) => (storage.embedding(idx), norms[idx]),
            WordIndex::Subword(indices) => {
                let mut embed = Array1::zeros((storage.shape().1,));
                for idx in indices {
                    embed += &storage.embedding(idx).view();
                }

                let norm = l2_normalize(embed.view_mut());

                (CowArray::from(embed), norm)
            }
        })
    }

    pub(crate) fn storage_(&self) -> Option<&StorageWrap> {
        self.storage.as_ref().map(|s| s.storage_())
    }

    pub(crate) fn vocab_(&self) -> Option<&VocabWrap> {
        self.vocab.as_ref().map(|v| v.vocab_())
    }

    pub(crate) fn norms_(&self) -> Option<&NdNorms> {
        self.norms.as_ref().map(|norms| norms.norms_())
    }

    fn set_storage(&mut self, storage: Option<PyStorage>) -> Option<PyStorage> {
        mem::replace(&mut self.storage, storage)
    }

    fn set_vocab(&mut self, vocab: Option<PyVocab>) -> Option<PyVocab> {
        mem::replace(&mut self.vocab, vocab)
    }

    fn set_norms(&mut self, norms: Option<PyNorms>) -> Option<PyNorms> {
        mem::replace(&mut self.norms, norms)
    }

    fn set_metadata(&mut self, metadata: Option<PyMetadata>) -> Option<PyMetadata> {
        mem::replace(&mut self.metadata, metadata)
    }

    pub(crate) fn header(&self) -> PyResult<Header> {
        let mut chunks = vec![];
        if self.metadata.is_some() {
            chunks.push(ChunkIdentifier::Metadata)
        }
        if let Some(vocab) = self.vocab.as_ref() {
            chunks.push(vocab.chunk_identifier())
        }
        if let Some(storage) = self.storage.as_ref() {
            chunks.push(storage.chunk_identifier())
        }
        if let Some(norms) = self.norms.as_ref() {
            chunks.push(norms.chunk_identifier())
        }
        if chunks.is_empty() {
            return Err(exceptions::TypeError::py_err(
                "Cannot serialize empty embeddings.",
            ));
        }
        Ok(Header::new(chunks))
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
        let f = File::open(path).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot open embeddings file for reading: {}\n{}",
                path, e
            ))
        })?;
        let mut reader = BufReader::new(f);
        let header = io::Header::read_chunk(&mut reader)?;
        let chunks = header.chunk_identifiers();
        if chunks.is_empty() {
            return Err(exceptions::IOError::py_err("File contains no chunks."));
        }
        let mut embeddings = PyEmbeddings::empty();
        use ChunkIdentifier::*;
        for chunk in chunks {
            match chunk {
                NdArray => {
                    embeddings.set_storage(Some(PyStorage::load_array(&mut reader, mmap)?));
                }
                QuantizedArray => {
                    embeddings.set_storage(Some(PyStorage::load_quantized(&mut reader, mmap)?));
                }
                NdNorms => {
                    embeddings.set_norms(Some(PyNorms::read_chunk(&mut reader)?));
                }
                Header => {
                    return Err(exceptions::IOError::py_err(
                        "File contains multiple headers.",
                    ))
                }
                SimpleVocab | BucketSubwordVocab | FastTextSubwordVocab | ExplicitSubwordVocab => {
                    embeddings.set_vocab(Some(PyVocab::read_chunk(&mut reader)?));
                }
                Metadata => {
                    embeddings.set_metadata(Some(PyMetadata::read_chunk(&mut reader)?));
                }
            };
        }
        if embeddings.norms.is_none() {
            embeddings.set_norms(PyNorms::read_chunk(&mut reader).ok());
        }

        obj.init(embeddings);

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
    fn vocab(&self) -> Option<PyVocab> {
        self.vocab.clone()
    }

    /// Get the model's storage.
    fn storage(&self) -> Option<PyStorage> {
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
        if self.vocab.is_none() {
            return Err(exceptions::TypeError::py_err(
                "Vocab is required for analogy queries.",
            ));
        };
        let storage = self.storage_().ok_or_else(|| {
            exceptions::TypeError::py_err("Storage is required for analogy queries.")
        })?;
        if storage.quantized() {
            return Err(exceptions::ValueError::py_err(
                "Analogy queries are not supported for this quantized embedding matrices",
            ));
        }

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
        query: PyQuery<&str>,
        default: PyEmbeddingDefault,
    ) -> PyResult<Option<Py<PyArray1<f32>>>> {
        if self.vocab.is_none() {
            return Err(exceptions::TypeError::py_err(
                "Vocab is required for embedding lookup.",
            ));
        }
        let (_, cols) = self.storage_().map(|s| s.shape()).ok_or_else(|| {
            exceptions::TypeError::py_err("Storage is required for embedding lookup.")
        })?;

        let gil = pyo3::Python::acquire_gil();
        if let PyEmbeddingDefault::Embedding(array) = &default {
            if array.as_ref(gil.python()).as_array().shape()[0] != cols {
                return Err(exceptions::ValueError::py_err(format!(
                    "Invalid shape of default embedding: {}",
                    array.as_ref(gil.python()).as_array().shape()[0]
                )));
            }
        }
        println!("{:?}", query);

        if let PyQuery::Word(word) = query {
            if let Some(embedding) = self.embedding_(&word) {
                let embedding = embedding.to_owned();

                return Ok(Some(
                    PyArray1::from_owned_array(gil.python(), embedding).to_owned(),
                ));
            };
        }
        match default {
            PyEmbeddingDefault::Constant(constant) => {
                let nd_arr = Array1::from_elem([cols], constant);
                Ok(Some(nd_arr.into_pyarray(gil.python()).to_owned()))
            }
            PyEmbeddingDefault::Embedding(array) => Ok(Some(array)),
            PyEmbeddingDefault::None => Ok(None),
        }
    }

    fn embedding_with_norm(&self, word: &str) -> PyResult<Option<Py<PyTuple>>> {
        if self.vocab.is_none() {
            return Err(exceptions::TypeError::py_err(
                "Vocab is required for embedding lookup.",
            ));
        }
        if self.storage.is_none() {
            return Err(exceptions::TypeError::py_err(
                "Storage is required for embedding lookup.",
            ));
        };

        if self.norms_().is_none() {
            return Err(exceptions::TypeError::py_err(
                "Norms are required for embedding lookup with norms.",
            ));
        }

        let embedding_with_norm = self.embedding_with_norm_(word);

        Ok(embedding_with_norm.map(|(embedding, norm)| {
            let gil = pyo3::Python::acquire_gil();
            let embedding = embedding.into_owned().into_pyarray(gil.python());
            (embedding, norm).into_py(gil.python())
        }))
    }

    /// Perform a similarity query.
    #[args(limit = 10)]
    fn word_similarity(&self, py: Python, word: &str, limit: usize) -> PyResult<Vec<PyObject>> {
        let storage = self.storage_();
        if storage.is_none() || self.vocab.is_none() {
            return Err(exceptions::TypeError::py_err(
                "Embeddings need to contain vocab and storage for similarity queries.",
            ));
        }

        if storage.unwrap().quantized() {
            return Err(exceptions::ValueError::py_err(
                "Similarity queries are not supported for this type of embedding matrix",
            ));
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
        let storage = self.storage_();
        if storage.is_none() || self.vocab.is_none() {
            return Err(exceptions::TypeError::py_err(
                "Embeddings need to contain vocab and storage for similarity queries.",
            ));
        }
        let storage = storage.unwrap();
        if storage.quantized() {
            return Err(exceptions::ValueError::py_err(
                "Similarity queries are not supported for this type of embedding matrix",
            ));
        };

        let embedding = embedding.0.as_array();
        if embedding.shape()[0] != storage.shape().1 {
            return Err(exceptions::ValueError::py_err(format!(
                "Incompatible embedding shapes: embeddings: ({},), query: ({},)",
                embedding.shape()[0],
                storage.shape().1
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
        let header = self.header()?;
        header.write_chunk(&mut writer)?;
        if let Some(metadata) = self.metadata.as_ref() {
            metadata.write_chunk(&mut writer)?;
        }
        if let Some(vocab) = self.vocab.as_ref() {
            vocab.write_chunk(&mut writer)?;
        }
        if let Some(storage) = self.storage.as_ref() {
            storage.write_chunk(&mut writer)?;
        }
        if let Some(norms) = self.norms.as_ref() {
            norms.write_chunk(&mut writer)?;
        }
        Ok(())
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

#[pyproto]
impl PyObjectProtocol for PyEmbeddings {
    fn __repr__(&self) -> PyResult<String> {
        let mut repr = "Embeddings {\n".to_string();
        if let Some(vocab) = self.vocab() {
            repr += "    vocab:";
            repr += &vocab.repr_("    vocab:".len());
            repr += ",\n";
        }
        if let Some(storage) = self.storage() {
            let storage_str = "    storage:";
            repr += storage_str;
            repr += &storage.repr_(storage_str.len());
            repr += ",\n";
        }
        if let Some(norms) = self.norms() {
            let norms_str = "    norms:";
            repr += norms_str;
            repr += &norms.repr_(norms_str.len());
            repr += ",\n";
        }
        if let Some(metadata) = self.metadata() {
            let metadata_str = "    metadata:";
            repr += metadata_str;
            repr += &metadata.repr_(metadata_str.len());
            repr += ",\n";
        }
        repr += "}";
        Ok(repr)
    }
}

fn read_non_fifu_embeddings<R, V>(path: &str, read_embeddings: R) -> PyResult<PyEmbeddings>
where
    R: FnOnce(&mut BufReader<File>) -> ffio::Result<Embeddings<V, NdArray>>,
    V: Vocab + Clone + Into<VocabWrap>,
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
    let storage = Some(PyStorage::new(Rc::new(storage.into())));
    let vocab = Some(PyVocab::new(Rc::new(vocab.into())));
    let norms = norms.map(|norms| PyNorms::new(Rc::new(norms)));
    let metadata = metadata.map(|metadata| PyMetadata::new(Rc::new(metadata)));

    Ok(PyEmbeddings {
        storage,
        vocab,
        norms,
        metadata,
    })
}
