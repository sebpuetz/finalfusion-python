use std::ops::Deref;
use std::rc::Rc;

use finalfusion::chunks::metadata::Metadata;
use pyo3::prelude::*;
use pyo3::{exceptions, PyObjectProtocol, PyResult};
use toml::Value;

/// finalfusion storage.
#[pyclass(name=Metadata)]
pub struct PyMetadata {
    metadata: Rc<Metadata>,
}

impl PyMetadata {
    pub fn new(metadata: Rc<Metadata>) -> Self {
        PyMetadata { metadata }
    }
}

impl Deref for PyMetadata {
    type Target = Metadata;

    fn deref(&self) -> &Self::Target {
        self.metadata.as_ref()
    }
}

#[pymethods]
impl PyMetadata {
    #[new]
    fn __new__(obj: &PyRawObject, metadata: &str) -> PyResult<()> {
        let value = match metadata.parse::<Value>() {
            Ok(value) => value,
            Err(err) => {
                return Err(exceptions::ValueError::py_err(format!(
                    "Metadata is invalid TOML: {}",
                    err
                )));
            }
        };
        obj.init(PyMetadata {
            metadata: Rc::new(Metadata(value)),
        });
        Ok(())
    }
}

#[pyproto]
impl<'a> PyObjectProtocol<'a> for PyMetadata {
    fn __str__(&self) -> PyResult<String> {
        Ok(toml::ser::to_string_pretty(&self.0)
            .map_err(|e| exceptions::IOError::py_err(format!("Metadata is invalid TOML: {}", e)))?)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }

}
