use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::mem::size_of;
use std::ops::Deref;
use std::rc::Rc;

use crate::io::{find_chunk, ChunkIdentifier, Header, ReadChunk, WriteChunk};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use finalfusion::chunks::vocab::{
    BucketSubwordVocab, ExplicitSubwordVocab, FastTextSubwordVocab, NGramIndices, SubwordIndices,
    WordIndex,
};
use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::prelude::*;
use finalfusion::subword::{
    BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, Indexer, NGrams,
};
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
    #[new]
    fn __new__(obj: &PyRawObject, filename: &str) -> PyResult<()> {
        let file = File::open(filename).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for reading: {}\n{}",
                filename, e
            ))
        })?;
        let mut reader = BufReader::new(file);
        let header = Header::read_chunk(&mut reader)?;
        let chunks = header.chunk_identifiers();
        for chunk in chunks {
            match chunk {
                ChunkIdentifier::SimpleVocab => {
                    obj.init(Self::read_simple_vocab(&mut reader)?);
                    return Ok(());
                }
                ChunkIdentifier::BucketSubwordVocab | ChunkIdentifier::FastTextSubwordVocab => {
                    obj.init(Self::read_bucketed_vocab(&mut reader)?);
                    return Ok(());
                }
                ChunkIdentifier::ExplicitSubwordVocab => {
                    obj.init(Self::read_explicit_vocab(&mut reader)?);
                    return Ok(());
                }
                _ => continue,
            }
        }
        Err(exceptions::IOError::py_err(
            "File did not contain a vocabulary.",
        ))
    }

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

    /// words(self,/,)
    /// --
    ///
    /// Get the list of words stored in this vocabulary.
    fn words(&self) -> Vec<String> {
        self.vocab.words().into()
    }

    /// ngrams(self,/,)
    /// --
    ///
    /// Get the list of ngrams explicitly stored in this vocabulary.
    fn ngrams(&self) -> PyResult<Vec<String>> {
        if let VocabWrap::ExplicitSubwordVocab(vocab) = self.vocab_() {
            Ok(vocab.indexer().ngrams().into())
        } else {
            Err(exceptions::TypeError::py_err(
                "this vocabulary type does not contain ngrams.",
            ))
        }
    }

    /// ngram_range(self,/,)
    /// --
    ///
    /// Return the lower and upper bound of the length of ngrams in this vocabulary.
    fn ngram_range(&self) -> PyResult<(u32, u32)> {
        Ok(match self.vocab_() {
            VocabWrap::ExplicitSubwordVocab(vocab) => (vocab.min_n(), vocab.max_n()),
            VocabWrap::BucketSubwordVocab(vocab) => (vocab.min_n(), vocab.max_n()),
            VocabWrap::FastTextSubwordVocab(vocab) => (vocab.min_n(), vocab.max_n()),
            VocabWrap::SimpleVocab(_) => {
                return Err(exceptions::TypeError::py_err(
                    "SimpleVocab does not index ngrams.",
                ))
            }
        })
    }

    /// index_ngram(self, ngram,/,)
    /// --
    ///
    /// Index the given ngram.
    fn index_ngram(&self, ngram: &str) -> PyResult<Option<u64>> {
        Ok(match self.vocab_() {
            VocabWrap::ExplicitSubwordVocab(vocab) => vocab.indexer().index_ngram(&ngram.into()),
            VocabWrap::BucketSubwordVocab(vocab) => vocab.indexer().index_ngram(&ngram.into()),
            VocabWrap::FastTextSubwordVocab(vocab) => vocab.indexer().index_ngram(&ngram.into()),
            VocabWrap::SimpleVocab(_) => {
                return Err(exceptions::TypeError::py_err(
                    "SimpleVocab does not index ngrams.",
                ))
            }
        })
    }

    /// word_ngrams(word,/, bracket, lower, upper)
    /// --
    ///
    /// Get the ngrams in `word`.
    ///
    /// `lower` and `upper` set the lower and upper bound for ngram lengths.
    /// `bracket` toggles bracketing the word with `'<'` and `'>'` before ngram extraction.
    ///
    /// **Note** finalfusion brackets tokens per default with `'<'` and `'>'`. When extracting
    /// ngrams, finalfusion vocabularies will always generate ngrams with these brackets.
    #[staticmethod]
    #[args(bracket = "true", lower = 3, upper = 6)]
    fn word_ngrams(word: &str, bracket: bool, lower: usize, upper: usize) -> PyResult<Vec<String>> {
        if lower >= upper || lower == 0 {
            return Err(exceptions::AssertionError::py_err(
                "'lower' needs to be nonzero integer and smaller than 'upper'",
            ));
        }

        if bracket {
            Ok(NGrams::new(&Self::bracket(word), lower, upper)
                .map(|s| s.to_string())
                .collect())
        } else {
            Ok(NGrams::new(&Self::bracket(word), lower, upper)
                .map(|s| s.to_string())
                .collect())
        }
    }

    /// full_len(self,/,)
    /// --
    ///
    /// Get the full length of this vocabulary, i.e. if this vocabulary indexes subwords,
    fn full_len(&self) -> usize {
        self.vocab.vocab_len()
    }

    /// write(self, filename,/,)
    /// --
    ///
    /// Write the vocabulary in finalfusion format to the given file.
    fn write(&self, filename: &str) -> PyResult<()> {
        let file = File::create(filename).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Could not open the file for writing: {}\n{}",
                filename, e
            ))
        })?;
        let mut writer = BufWriter::new(file);
        let header = Header::new(vec![self.into()]);
        header.write_chunk(&mut writer)?;
        self.write_chunk(&mut writer)
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

impl<'a> From<&'a PyVocab> for ChunkIdentifier {
    fn from(vocab: &'a PyVocab) -> Self {
        use VocabWrap::*;
        match vocab.vocab_() {
            SimpleVocab(_) => ChunkIdentifier::SimpleVocab,
            BucketSubwordVocab(_) => ChunkIdentifier::BucketSubwordVocab,
            FastTextSubwordVocab(_) => ChunkIdentifier::FastTextSubwordVocab,
            ExplicitSubwordVocab(_) => ChunkIdentifier::ExplicitSubwordVocab,
        }
    }
}

impl WriteChunk for PyVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        self.into()
    }

    fn write_chunk<W>(&self, write: &mut W) -> PyResult<()>
    where
        W: Write,
    {
        use VocabWrap::*;
        match self.vocab_() {
            BucketSubwordVocab(vocab) => {
                Self::write_bucketed_vocab(write, vocab, ChunkIdentifier::BucketSubwordVocab)
            }
            SimpleVocab(vocab) => Self::write_simple_vocab(write, vocab),
            ExplicitSubwordVocab(vocab) => Self::write_ngram_chunk(vocab, write),
            FastTextSubwordVocab(vocab) => {
                Self::write_bucketed_vocab(write, vocab, ChunkIdentifier::FastTextSubwordVocab)
            }
        }
    }
}

impl PyVocab {
    fn read_simple_vocab<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        find_chunk(read, &[ChunkIdentifier::SimpleVocab])?;
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::SimpleVocab)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read vocabulary chunk length\n{}", e))
        })?;

        let vocab_len = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read vocabulary length\n{}", e))
        })? as usize;

        let words = read_vocab_items(read, vocab_len)?;

        Ok(PyVocab::new(Rc::new(SimpleVocab::new(words).into())))
    }

    fn write_simple_vocab<W>(write: &mut W, vocab: &SimpleVocab) -> PyResult<()>
    where
        W: Write,
    {
        let chunk_len = size_of::<u64>()
            + vocab
                .words()
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(ChunkIdentifier::SimpleVocab as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write vocabulary chunk identifier\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write vocabulary chunk length\n{}", e))
            })?;
        write
            .write_u64::<LittleEndian>(vocab.words().len() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write vocabulary length\n{}", e))
            })?;

        write_vocab_items(write, vocab.words())?;

        Ok(())
    }

    fn read_bucketed_vocab<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        let identifier = find_chunk(
            read,
            &[
                ChunkIdentifier::BucketSubwordVocab,
                ChunkIdentifier::BucketSubwordVocab,
            ],
        )?;
        ChunkIdentifier::ensure_chunk_type(read, identifier)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read vocabulary chunk length\n{}", e))
        })?;

        let vocab_len = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read vocabulary length\n{}", e))
        })? as usize;
        let min_n = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read minimum n-gram length\n{}", e))
        })?;
        let max_n = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read maximum n-gram length\n{}", e))
        })?;
        let buckets = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read number of buckets\n{}", e))
        })?;

        let words = read_vocab_items(read, vocab_len as usize)?;
        match identifier {
            ChunkIdentifier::BucketSubwordVocab => {
                let indexer = FinalfusionHashIndexer::new(buckets as usize);
                let vocab = BucketSubwordVocab::new(words, min_n, max_n, indexer).into();
                Ok(PyVocab::new(Rc::new(vocab)))
            }
            ChunkIdentifier::FastTextSubwordVocab => {
                let indexer = FastTextIndexer::new(buckets as usize);
                let vocab = FastTextSubwordVocab::new(words, min_n, max_n, indexer).into();
                Ok(PyVocab::new(Rc::new(vocab)))
            }
            id => Err(exceptions::ValueError::py_err(format!(
                "Expected one of [{}, {}], but got {}",
                ChunkIdentifier::BucketSubwordVocab,
                ChunkIdentifier::FastTextSubwordVocab,
                id
            ))),
        }
    }

    fn write_bucketed_vocab<I, W>(
        write: &mut W,
        vocab: &SubwordVocab<I>,
        identifier: ChunkIdentifier,
    ) -> PyResult<()>
    where
        I: BucketIndexer,
        W: Write,
    {
        // Chunk size: vocab size (u64), minimum n-gram length (u32),
        // maximum n-gram length (u32), bucket exponent (u32), for
        // each word: word length in bytes (u32), word bytes
        // (variable-length).
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + vocab
                .words()
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(identifier as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write subword vocabulary chunk identifier\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write subword vocabulary chunk length\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(vocab.words().len() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write vocabulary length\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(vocab.min_n())
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write minimum n-gram length\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(vocab.max_n())
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write maximum n-gram length\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(vocab.indexer().buckets() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write number of buckets\n{}", e))
            })?;

        write_vocab_items(write, vocab.words())?;

        Ok(())
    }

    fn read_explicit_vocab<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        let identifier = find_chunk(read, &[ChunkIdentifier::ExplicitSubwordVocab])?;
        ChunkIdentifier::ensure_chunk_type(read, identifier)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read vocabulary chunk length\n{}", e))
        })?;

        let words_len = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read vocabulary length\n{}", e))
        })? as usize;
        let ngrams_len = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read number of ngrams\n{}", e))
        })?;
        let min_n = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read minimum n-gram length\n{}", e))
        })?;
        let max_n = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read maximum n-gram length\n{}", e))
        })?;

        let words = read_vocab_items(read, words_len as usize)?;
        let ngrams = read_vocab_items(read, ngrams_len as usize)?;
        let indexer = ExplicitIndexer::new(ngrams);
        let vocab = ExplicitSubwordVocab::new(words, min_n, max_n, indexer).into();
        Ok(PyVocab::new(Rc::new(vocab)))
    }

    fn write_ngram_chunk<W>(vocab: &ExplicitSubwordVocab, write: &mut W) -> PyResult<()>
    where
        W: Write,
    {
        // Chunk size: word vocab size (u64), ngram vocab size (u64)
        // minimum n-gram length (u32), maximum n-gram length (u32),
        // for each word and ngram:
        // length in bytes (u32), number of bytes (variable-length).
        let chunk_len = size_of::<u64>()
            + size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + vocab
                .words()
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>()
            + vocab
                .indexer()
                .ngrams()
                .iter()
                .map(|ngram| ngram.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(ChunkIdentifier::ExplicitSubwordVocab as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write subword vocabulary chunk identifier\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write subword vocabulary chunk length\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(vocab.words().len() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write vocabulary length\n{}", e))
            })?;
        write
            .write_u64::<LittleEndian>(vocab.indexer().ngrams().len() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write ngram length\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(vocab.min_n())
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write minimum n-gram length\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(vocab.max_n())
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write maximum n-gram length\n{}", e))
            })?;

        write_vocab_items(write, vocab.words())?;
        write_vocab_items(write, vocab.indexer().ngrams())?;

        Ok(())
    }
}

fn read_vocab_items<R>(read: &mut R, len: usize) -> PyResult<Vec<String>>
where
    R: Read,
{
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        let item_len = read
            .read_u32::<LittleEndian>()
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot read item length\n{}", e)))?
            as usize;
        let mut bytes = vec![0; item_len];
        read.read_exact(&mut bytes)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot read item\n{}", e)))?;
        let item = String::from_utf8(bytes).map_err(|e| {
            exceptions::IOError::py_err(format!("Item contains invalid UTF-8: {}", e))
        })?;
        items.push(item);
    }
    Ok(items)
}

fn write_vocab_items<W>(write: &mut W, items: &[String]) -> PyResult<()>
where
    W: Write,
{
    for word in items {
        write
            .write_u32::<LittleEndian>(word.len() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write token length\n{}", e))
            })?;
        write
            .write_all(word.as_bytes())
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write token\n{}", e)))?;
    }
    Ok(())
}
