use std::collections::{BinaryHeap, HashSet};

use finalfusion::similarity::*;
use ndarray::{s, Array1, ArrayView1, ArrayView2, CowArray, Ix1};
use ordered_float::NotNan;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::prelude::*;

use finalfusion::storage::StorageView;
use finalfusion::vocab::Vocab;

use crate::embeddings::PyEmbeddings;
use crate::storage::StorageWrap;
use crate::util::l2_normalize;

impl Analogy for PyEmbeddings {
    fn analogy_masked(
        &self,
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]> {
        self.analogy_by_masked(query, remove, limit, |embeds, embed| embeds.dot(&embed))
    }
}

impl AnalogyBy for PyEmbeddings {
    fn analogy_by_masked<F>(
        &self,
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
        similarity: F,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let [embedding1, embedding2, embedding3] = lookup_words3(self, query)?;

        let mut embedding = (&embedding2.view() - &embedding1.view()) + embedding3.view();
        l2_normalize(embedding.view_mut());

        let skip = query
            .iter()
            .zip(remove.iter())
            .filter(|(_, &exclude)| exclude)
            .map(|(word, _)| word.to_owned())
            .collect();
        Ok(self.similarity_(embedding.view(), &skip, limit, similarity))
    }
}

impl WordSimilarity for PyEmbeddings {
    fn word_similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarityResult>> {
        self.word_similarity_by(word, limit, |embeds, embed| embeds.dot(&embed))
    }
}

impl WordSimilarityBy for PyEmbeddings {
    fn word_similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarityResult>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embed = self.embedding_(word)?;
        let mut skip = HashSet::new();
        skip.insert(word);

        Some(self.similarity_(embed.view(), &skip, limit, similarity))
    }
}

impl EmbeddingSimilarity for PyEmbeddings {
    fn embedding_similarity_masked(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skips: &HashSet<&str>,
    ) -> Option<Vec<WordSimilarityResult>> {
        self.embedding_similarity_by(query, limit, skips, |embeds, embed| embeds.dot(&embed))
    }
}

impl EmbeddingSimilarityBy for PyEmbeddings {
    fn embedding_similarity_by<F>(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skips: &HashSet<&str>,
        similarity: F,
    ) -> Option<Vec<WordSimilarityResult>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        Some(self.similarity_(query, skips, limit, similarity))
    }
}

/// A word and its similarity to a query word.
///
/// The similarity is normally a value between -1 (opposite
/// vectors) and 1 (identical vectors).
#[pyclass(name=WordSimilarity)]
pub struct PyWordSimilarity {
    #[pyo3(get)]
    word: String,

    #[pyo3(get)]
    similarity: f32,
}

impl PyWordSimilarity {
    pub fn new(word: String, similarity: f32) -> Self {
        PyWordSimilarity { word, similarity }
    }
}

#[pyproto]
impl PyObjectProtocol for PyWordSimilarity {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "WordSimilarity('{}', {})",
            self.word, self.similarity
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}: {}", self.word, self.similarity))
    }
}

pub(crate) fn similarity_results(
    py: Python,
    results: Vec<WordSimilarityResult>,
) -> PyResult<Vec<PyObject>> {
    let mut r = Vec::with_capacity(results.len());
    for ws in results {
        r.push(IntoPy::into_py(
            Py::new(
                py,
                PyWordSimilarity::new(ws.word.to_owned(), ws.similarity.into_inner()),
            )?,
            py,
        ))
    }
    Ok(r)
}

trait SimilarityPrivate {
    fn similarity_<F>(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        similarity: F,
    ) -> Vec<WordSimilarityResult>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl SimilarityPrivate for PyEmbeddings {
    fn similarity_<F>(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        mut similarity: F,
    ) -> Vec<WordSimilarityResult>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let storage = self
            .storage_()
            .expect("Storage is required for similarity queries.");
        let vocab = self
            .vocab_()
            .expect("Storage is required for similarity queries.");
        use StorageWrap::*;
        let view = match storage {
            MmapQuantizedArray(_) | QuantizedArray(_) => {
                unreachable!("This similarity fn should not be reachable.")
            }
            MmapArray(array) => array.view(),
            NdArray(array) => array.view(),
        };

        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sims = similarity(view.slice(s![0..vocab.words_len(), ..]), embed.view());

        let mut results = BinaryHeap::with_capacity(limit);
        for (idx, &sim) in sims.iter().enumerate() {
            let word = &vocab.words()[idx];

            // Don't add words that we are explicitly asked to skip.
            if skip.contains(word.as_str()) {
                continue;
            }

            let word_similarity = WordSimilarityResult {
                word,
                similarity: NotNan::new(sim).expect("Encountered NaN"),
            };

            if results.len() < limit {
                results.push(word_similarity);
            } else {
                let mut peek = results.peek_mut().expect("Cannot peek non-empty heap");
                if word_similarity < *peek {
                    *peek = word_similarity
                }
            }
        }

        results.into_sorted_vec()
    }
}

fn lookup_words3<'a>(
    embeddings: &'a PyEmbeddings,
    query: [&str; 3],
) -> Result<[CowArray<'a, f32, Ix1>; 3], [bool; 3]> {
    let embedding1 = embeddings.embedding_(query[0]);
    let embedding2 = embeddings.embedding_(query[1]);
    let embedding3 = embeddings.embedding_(query[2]);

    let present = [
        embedding1.is_some(),
        embedding2.is_some(),
        embedding3.is_some(),
    ];

    if !present.iter().all(|&present| present) {
        return Err(present);
    }

    Ok([
        embedding1.unwrap(),
        embedding2.unwrap(),
        embedding3.unwrap(),
    ])
}
