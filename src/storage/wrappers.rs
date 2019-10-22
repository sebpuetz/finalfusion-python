use crate::storage::quantized::MmapQuantizedArray;
use crate::storage::{MmapArray, QuantizedArray};
use finalfusion::storage::{NdArray, Storage};
use ndarray::{CowArray, Ix1};

#[allow(dead_code)]
pub enum StorageWrap {
    NdArray(NdArray),
    MmapArray(MmapArray),
    QuantizedArray(Box<QuantizedArray>),
    MmapQuantizedArray(MmapQuantizedArray),
}

impl StorageWrap {
    pub fn quantized(&self) -> bool {
        match self {
            StorageWrap::QuantizedArray(_) | StorageWrap::MmapQuantizedArray(_) => true,
            _ => false,
        }
    }
}

impl Storage for StorageWrap {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        match self {
            StorageWrap::NdArray(array) => array.embedding(idx),
            StorageWrap::MmapArray(array) => array.embedding(idx),
            StorageWrap::QuantizedArray(array) => array.embedding(idx),
            StorageWrap::MmapQuantizedArray(array) => array.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            StorageWrap::NdArray(array) => array.shape(),
            StorageWrap::MmapArray(array) => array.shape(),
            StorageWrap::QuantizedArray(array) => array.shape(),
            StorageWrap::MmapQuantizedArray(array) => array.shape(),
        }
    }
}

impl From<MmapArray> for StorageWrap {
    fn from(s: MmapArray) -> Self {
        StorageWrap::MmapArray(s)
    }
}

impl From<NdArray> for StorageWrap {
    fn from(s: NdArray) -> Self {
        StorageWrap::NdArray(s)
    }
}
