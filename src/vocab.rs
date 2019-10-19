use std::collections::HashSet;
use std::ops::Deref;
use std::rc::Rc;

use finalfusion::chunks::vocab::{
    BucketSubwordVocab, FastTextSubwordVocab, NGramIndices, SubwordIndices, WordIndex,
};
use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::prelude::*;
use finalfusion::subword::{BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, NGrams};
use itertools::Itertools;
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::prelude::*;
use pyo3::{exceptions, PyObjectProtocol};

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

    pub fn bracket(word: &str) -> String {
        let mut bracketed = String::with_capacity(word.len() + 2);
        bracketed.push('<');
        bracketed.push_str(word.as_ref());
        bracketed.push('>');
        bracketed
    }
}

fn check_duplicates(l: &[String], msg: impl Into<String>) -> PyResult<()> {
    let unique = l.iter().collect::<HashSet<_>>().len();
    if unique != l.len() {
        Err(exceptions::RuntimeError::py_err(msg.into()))
    } else {
        Ok(())
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
    fn simple_vocab(words: Vec<String>) -> PyResult<Self> {
        check_duplicates(&words, "'words' cannot contain duplicate entries")?;
        Ok(PyVocab::new(Rc::new(SimpleVocab::new(words).into())))
    }

    /// ff_bucket_vocab(words,/, min_n, max_n, buckets_exp)
    /// --
    ///
    /// Construct a Finalfusion bucket vocab. Indices are assigned according to the given order.
    #[staticmethod]
    #[args(min_n = 3, max_n = 6, buckets_exp = 21)]
    fn ff_bucket_vocab(
        words: Vec<String>,
        min_n: u32,
        max_n: u32,
        buckets_exp: usize,
    ) -> PyResult<Self> {
        check_duplicates(&words, "'words' cannot contain duplicate entries")?;
        let indexer = FinalfusionHashIndexer::new(buckets_exp);
        let vocab = BucketSubwordVocab::new(words, min_n, max_n, indexer);
        Ok(PyVocab::new(Rc::new(vocab.into())))
    }

    /// fasttext_vocab(words,/, min_n, max_n, buckets_exp)
    /// --
    ///
    /// Construct a fastText vocab. Indices are assigned according to the given order.
    #[staticmethod]
    #[args(min_n = 3, max_n = 6, n_buckets = 2000000)]
    fn fasttext_vocab(
        words: Vec<String>,
        min_n: u32,
        max_n: u32,
        n_buckets: usize,
    ) -> PyResult<Self> {
        check_duplicates(&words, "'words' cannot contain duplicate entries")?;
        let vocab = FastTextSubwordVocab::new(words, min_n, max_n, FastTextIndexer::new(n_buckets));
        Ok(PyVocab::new(Rc::new(vocab.into())))
    }

    /// ff_ngram_vocab(words,/, ngrams, min_n, max_n)
    /// --
    ///
    /// Construct a Finalfusion vocab with explicit subword lookup.
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
    fn ff_explicit_vocab(
        words: Vec<String>,
        mut ngrams: Vec<String>,
        min_n: usize,
        max_n: usize,
    ) -> PyResult<Self> {
        check_duplicates(&words, "'words' cannot contain duplicate entries")?;
        if ngrams.is_empty() {
            let mut ngram_set = HashSet::new();
            if min_n * max_n == 0 {
                return Err(exceptions::ValueError::py_err(
                    "min_n and max_n can't be zero without explicitly given ngrams.",
                ));
            }
            for word in words.iter() {
                let word = Self::bracket(word);
                let ngram_iter = NGrams::new(&word, min_n, max_n).filter_map(|ngram| {
                    if !ngram_set.contains(ngram.as_str()) {
                        ngram_set.insert(ngram.to_string());
                        Some(ngram.to_string())
                    } else {
                        None
                    }
                });
                ngrams.extend(ngram_iter);
            }
        } else {
            check_duplicates(&ngrams, "'ngrams' cannot contain duplicate entries")?;
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
        let indexer = ExplicitIndexer::new(ngrams);
        let vocab = SubwordVocab::new(words, min_n, max_n, indexer);
        Ok(PyVocab {
            vocab: Rc::new(vocab.into()),
        })
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
            VocabWrap::BucketSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.ngram_indices(word),
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
            VocabWrap::BucketSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::ExplicitSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::SimpleVocab(_) => Err(exceptions::ValueError::py_err(
                "querying subwords' indices is not supported for this vocabulary",
            )),
        }
    }
}

#[pyproto]
impl PyObjectProtocol for PyVocab {
    fn __repr__(&self) -> PyResult<String> {
        let mut repr = String::new();
        match self.vocab_() {
            VocabWrap::BucketSubwordVocab(vocab) => {
                repr.push_str("BucketSubwordVocab {\n");
                repr.push_str(&format!("\tmin_n: {},\n", vocab.min_n()));
                repr.push_str(&format!("\tmax_n: {},\n", vocab.max_n()));
                repr.push_str(&format!("\tbuckets_exp: {},\n", vocab.indexer().buckets()));
                repr.push_str(&format_index(&vocab.words(), "words"));
            }
            VocabWrap::FastTextSubwordVocab(vocab) => {
                repr.push_str("FastTextVocab {\n");
                repr.push_str(&format!("\tmin_n: {},\n", vocab.min_n()));
                repr.push_str(&format!("\tmax_n: {},\n", vocab.max_n()));
                repr.push_str(&format!("\tn_buckets: {},\n", vocab.indexer().buckets()));
                repr.push_str(&format_index(&vocab.words(), "words"));
            }
            VocabWrap::SimpleVocab(vocab) => {
                repr.push_str("SimpleVocab {\n");
                repr.push_str(&format_index(&vocab.words(), "words"));
            }
            VocabWrap::ExplicitSubwordVocab(vocab) => {
                repr.push_str("ExplicitSubwordVocab {\n");
                repr.push_str(&format!("\tmin_n: {},\n", vocab.min_n()));
                repr.push_str(&format!("\tmax_n: {},\n", vocab.max_n()));
                repr.push_str(&format_index(&vocab.words(), "words"));
                repr.push_str(&format_index(&vocab.indexer().ngrams(), "ngrams"));
            }
        }
        repr.push('}');
        Ok(repr)
    }
}

fn format_index(words: &[String], name: &str) -> String {
    if words.len() > 10 {
        format!(
            "\t{}: {{{},...}},\n",
            name,
            words
                .iter()
                .enumerate()
                .take(10)
                .map(|(idx, word)| format!("'{}': {}", word, idx))
                .join(", ")
        )
    } else {
        format!(
            "\t{}: {{{}}},\n",
            name,
            words
                .iter()
                .enumerate()
                .map(|(idx, word)| format!("'{}': {}", word, idx))
                .join(", ")
        )
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
