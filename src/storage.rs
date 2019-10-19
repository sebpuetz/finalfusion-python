use std::rc::Rc;

use finalfusion::prelude::{Storage, StorageView, StorageWrap};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;

use std::ops::Deref;

/// finalfusion storage.
#[pyclass(name=Storage)]
#[derive(Clone)]
pub struct PyStorage {
    storage: Rc<StorageWrap>,
}

impl PyStorage {
    pub fn storage_(&self) -> &StorageWrap {
        self.storage.as_ref()
    }
}

impl Deref for PyStorage {
    type Target = StorageWrap;

    fn deref(&self) -> &Self::Target {
        self.storage.as_ref()
    }
}

impl PyStorage {
    pub fn new(storage: Rc<StorageWrap>) -> Self {
        PyStorage { storage }
    }
    /// Copy storage to an array.
    ///
    /// This should only be used for storage types that do not provide
    /// an ndarray view that can be copied trivially, such as quantized
    /// storage.
    pub(crate) fn copy_storage_to_array(storage: &dyn Storage) -> Array2<f32> {
        let (rows, dims) = storage.shape();

        let mut array = Array2::<f32>::zeros((rows, dims));
        for idx in 0..rows {
            array.row_mut(idx).assign(&storage.embedding(idx));
        }

        array
    }
}

#[pymethods]
impl PyStorage {
    /// Copy the entire embeddings matrix.
    pub fn matrix_copy(&self) -> Py<PyArray2<f32>> {
        let gil = pyo3::Python::acquire_gil();
        let matrix_view = match self.storage.as_ref() {
            StorageWrap::MmapArray(array) => array.view(),
            StorageWrap::NdArray(array) => array.view(),
            StorageWrap::QuantizedArray(quantized) => {
                let array = Self::copy_storage_to_array(quantized.as_ref());
                return array.to_pyarray(gil.python()).to_owned();
            }
            StorageWrap::MmapQuantizedArray(quantized) => {
                let array = Self::copy_storage_to_array(quantized);
                return array.to_pyarray(gil.python()).to_owned();
            }
        };

        matrix_view.to_pyarray(gil.python()).to_owned()
    }

    /// Get the shape of the storage.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.storage.shape()
    }
}

#[pyproto]
impl PySequenceProtocol for PyStorage {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.shape().0)
    }

    fn __getitem__(&self, idx: isize) -> PyResult<Py<PyArray1<f32>>> {
        if idx >= self.shape().0 as isize || idx < 0 {
            Err(exceptions::IndexError::py_err("list index out of range"))
        } else {
            let gil = Python::acquire_gil();
            Ok(self
                .embedding(idx as usize)
                .into_owned()
                .to_pyarray(gil.python())
                .to_owned())
        }
    }
}
