use std::ops::Deref;
use std::rc::Rc;

use finalfusion::chunks::norms::NdNorms;
use pyo3::prelude::*;
use pyo3::{exceptions, PyResult, PySequenceProtocol};

/// finalfusion storage.
#[pyclass(name=Norms)]
#[derive(Clone)]
pub struct PyNorms {
    norms: Rc<NdNorms>,
}

impl PyNorms {
    pub fn new(norms: Rc<NdNorms>) -> Self {
        PyNorms { norms }
    }

    pub(crate) fn norms_(&self) -> &NdNorms {
        self.norms.as_ref()
    }
}

impl Deref for PyNorms {
    type Target = NdNorms;

    fn deref(&self) -> &Self::Target {
        self.norms.as_ref()
    }
}

#[pyproto]
impl<'a> PySequenceProtocol<'a> for PyNorms {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }

    fn __getitem__(&self, idx: isize) -> PyResult<f32> {
        if idx >= self.0.len() as isize || idx < 0 {
            Err(exceptions::IndexError::py_err("list index out of range"))
        } else {
            Ok(self.0[idx as usize])
        }
    }
}
