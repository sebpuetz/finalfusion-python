use std::ops::Deref;
use std::rc::Rc;

use finalfusion::chunks::vocab::{
    FastTextSubwordVocab, FinalfusionSubwordVocab, NGramIndices, SubwordIndices, VocabWrap,
    WordIndex,
};
use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::prelude::*;
use finalfusion::subword::{BucketIndexer, FinalfusionHashIndexer, Indexer, NGramIndexer, NGrams};
use itertools::Itertools;
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyAny;

type NGramIndex = (String, Option<usize>);

/// finalfusion vocab.
#[pyclass(name = Vocab)]
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
    pub fn new(vocab: VocabWrap) -> Self {
        PyVocab {
            vocab: Rc::new(vocab),
        }
    }

    pub fn clone_vocab(&self) -> VocabWrap {
        self.vocab.as_ref().clone()
    }

    pub fn bracket(word: &str) -> String {
        let mut bracketed = String::with_capacity(word.len() + 2);
        bracketed.push('<');
        bracketed.push_str(word.as_ref());
        bracketed.push('>');
        bracketed
    }
}

#[pymethods]
impl PyVocab {
    /// simple_vocab(words,/)
    /// --
    ///
    /// Construct a SimpleVocab from the given words. Indices are assigned according to the given
    /// order.
    #[staticmethod]
    fn simple_vocab(words: Vec<String>) -> Self {
        PyVocab {
            vocab: Rc::new(VocabWrap::SimpleVocab(SimpleVocab::new(words))),
        }
    }

    /// ff_bucket_vocab(words,/, min_n, max_n, buckets_exp)
    /// --
    ///
    /// Construct a Finalfusion bucket vocab. Indices are assigned according to the given order.
    #[staticmethod]
    #[args(min_n = 3, max_n = 6, buckets_exp = 21)]
    fn ff_bucket_vocab(words: Vec<String>, min_n: u32, max_n: u32, buckets_exp: usize) -> Self {
        let vocab = FinalfusionSubwordVocab::new(
            words,
            min_n,
            max_n,
            FinalfusionHashIndexer::new(buckets_exp),
        );
        PyVocab {
            vocab: Rc::new(VocabWrap::FinalfusionSubwordVocab(vocab)),
        }
    }

    /// fasttext_vocab(words,/, min_n, max_n, buckets_exp)
    /// --
    ///
    /// Construct a fastText vocab. Indices are assigned according to the given order.
    #[staticmethod]
    #[args(min_n = 3, max_n = 6, n_buckets = 2000000)]
    fn fasttext_vocab(words: Vec<String>, min_n: u32, max_n: u32, n_buckets: usize) -> Self {
        let vocab = FastTextSubwordVocab::new(words, min_n, max_n, FastTextIndexer::new(n_buckets));
        PyVocab {
            vocab: Rc::new(VocabWrap::FastTextSubwordVocab(vocab)),
        }
    }

    /// ff_ngram_vocab(words,/, ngrams, min_n, max_n)
    /// --
    ///
    /// Construct a Finalfusion ngram vocab.
    ///
    /// If `ngrams` is empty or not given as an argument, ngrams are extracted from `words`. Based
    /// on `min_n` and `max_n`. If `min_n` or `max_n` are set to `0` without passing `ngrams`,
    /// an exception is raised.
    ///
    /// Otherwise, if any of `min_n` and `max_n` equal `0`, their corresponding value will be
    /// inferred from the lengths of the ngrams in `ngrams`.
    ///
    /// **Note**: Finalfusion brackets ngrams with "<" and ">", when generating ngrams for a word,
    /// the indexer will always generate ngrams containing these. `word_ngrams()` brackets words
    /// per default. If `ngrams' is not given as an argument, each word is bracketed before
    /// extracting its ngrams (this does not add brackets to the in-vocab tokens).
    #[staticmethod]
    #[args(ngrams = "Vec::new()", min_n = 3, max_n = 6)]
    fn ff_ngram_vocab(
        words: Vec<String>,
        mut ngrams: Vec<String>,
        min_n: usize,
        max_n: usize,
    ) -> PyResult<Self> {
        if ngrams.is_empty() {
            if min_n * max_n == 0 {
                return Err(exceptions::ValueError::py_err(
                    "min_n and max_n can't be zero without explicitly given ngrams.",
                ));
            }
            for word in words.iter() {
                let word = Self::bracket(word);
                ngrams.extend(NGrams::new(&word, min_n, max_n).map(|s| s.to_string()));
            }
        };
        let (min_n, max_n) = if min_n * max_n == 0 {
            ngrams
                .iter()
                .map(|s| s.len() as u32)
                .minmax()
                .into_option()
                .unwrap()
        } else {
            (min_n as u32, max_n as u32)
        };

        let vocab = VocabWrap::FinalfusionNGramVocab(SubwordVocab::new(
            words,
            min_n,
            max_n,
            NGramIndexer::new(ngrams),
        ));
        Ok(PyVocab {
            vocab: Rc::new(vocab),
        })
    }

    /// Get the ngrams stored in this vocabulary.
    fn ngrams(&self) -> Option<Vec<String>> {
        if let VocabWrap::FinalfusionNGramVocab(vocab) = self.vocab.as_ref() {
            Some(vocab.indexer().ngrams().to_vec())
        } else {
            None
        }
    }

    /// ngrams(word,/ min_n, max_n, bracket)
    /// --
    ///
    /// Get the list of ngrams in this word.
    ///
    /// **Note:** finalfusion subword vocabs bracket words with "<" and ">" before extracting
    /// ngrams. `bracket` toggles the bracketing. If you intend to get ngrams without added
    /// brackets, set this parameter to `False`.
    #[staticmethod]
    #[args(min_n = 3, max_n = 6, bracket = true)]
    fn word_ngrams(word: &str, min_n: usize, max_n: usize, bracket: bool) -> Vec<String> {
        let word = if bracket {
            Self::bracket(word)
        } else {
            word.to_string()
        };

        NGrams::new(&word, min_n, max_n)
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the subword index for the given ngram.
    fn ngram_idx(&self, ngram: &str) -> PyResult<Option<u64>> {
        Ok(match self.vocab.as_ref() {
            VocabWrap::FastTextSubwordVocab(inner) => inner.indexer().index_ngram(&ngram.into()),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.indexer().index_ngram(&ngram.into()),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.indexer().index_ngram(&ngram.into()),
            VocabWrap::SimpleVocab(_) => {
                return Err(exceptions::TypeError::py_err(
                    "querying n-gram indices is not supported for this vocabulary",
                ));
            }
        }
        .map(|idx| idx + self.vocab.words_len() as u64))
    }

    /// Get the subword index for the given ngram.
    fn word_idx(&self, word: &str) -> Option<usize> {
        self.vocab.as_ref().idx(word).and_then(|idx| idx.word())
    }

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

    fn max_idx(&self) -> usize {
        self.vocab.vocab_len()
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

impl<'a> FromPyObject<'a> for PyVocab {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let vocab = ob.downcast_ref::<PyVocab>()?;
        Ok(vocab.clone())
    }
}
