#![feature(specialization)]

mod embeddings;
use embeddings::Embeddings;

use pyo3::prelude::*;

#[pymodule]
fn finalfusion_standalone(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Embeddings>().unwrap();
    Ok(())
}
