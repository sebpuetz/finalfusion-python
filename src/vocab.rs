use std::ops::Deref;
use std::rc::Rc;

use finalfusion::chunks::vocab::{NGramIndices, SubwordIndices, WordIndex};
use finalfusion::prelude::*;
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;

type NGramIndex = (String, Option<usize>);

/// finalfusion vocab.
#[pyclass(name=Vocab)]
#[derive(Clone)]
pub struct PyVocab {
    vocab: Rc<VocabWrap>,
}

impl Deref for PyVocab {
    type Target = VocabWrap;

    fn deref(&self) -> &Self::Target {
        self.vocab.as_ref()
    }
}

impl PyVocab {
    pub fn new(vocab: Rc<VocabWrap>) -> Self {
        PyVocab { vocab }
    }

    pub(crate) fn vocab_(&self) -> &VocabWrap {
        self.vocab.as_ref()
    }
}

#[pymethods]
impl PyVocab {
    fn item_to_indices(&self, key: String) -> Option<PyObject> {
        self.vocab.idx(key.as_str()).map(|idx| {
            let gil = pyo3::Python::acquire_gil();
            match idx {
                WordIndex::Word(idx) => [idx].to_object(gil.python()),
                WordIndex::Subword(indices) => indices.to_object(gil.python()),
            }
        })
    }

    fn ngram_indices(&self, word: &str) -> PyResult<Option<Vec<NGramIndex>>> {
        Ok(match self.vocab.as_ref() {
            VocabWrap::FastTextSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.ngram_indices(word),
            VocabWrap::SimpleVocab(_) => {
                return Err(exceptions::ValueError::py_err(
                    "querying n-gram indices is not supported for this vocabulary",
                ));
            }
        })
    }

    fn subword_indices(&self, word: &str) -> PyResult<Option<Vec<usize>>> {
        match self.vocab.as_ref() {
            VocabWrap::FastTextSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::FinalfusionSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::FinalfusionNGramVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::SimpleVocab(_) => Err(exceptions::ValueError::py_err(
                "querying subwords' indices is not supported for this vocabulary",
            )),
        }
    }
}

#[pyproto]
impl PySequenceProtocol for PyVocab {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.vocab.words_len())
    }

    fn __getitem__(&self, idx: isize) -> PyResult<String> {
        let words = self.vocab.words();

        if idx >= words.len() as isize || idx < 0 {
            Err(exceptions::IndexError::py_err("list index out of range"))
        } else {
            Ok(words[idx as usize].clone())
        }
    }

    fn __contains__(&self, word: String) -> PyResult<bool> {
        Ok(self
            .vocab
            .idx(&word)
            .and_then(|word_idx| word_idx.word())
            .is_some())
    }
}
