use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use finalfusion::storage::{Storage, StorageView};
use memmap::Mmap;
use ndarray::{
    Array, Array1, Array2, ArrayView1, ArrayView2, CowArray, Dimension, IntoDimension, Ix1,
};
use pyo3::exceptions;
use pyo3::prelude::*;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use reductive::pq::{QuantizeVector, ReconstructVector, TrainPQ, PQ};

use crate::io::{padding, ChunkIdentifier, TypeId};
use crate::storage::{PyStorage, StorageWrap};
use crate::util::mmap_array;

/// Quantized embedding matrix.
pub struct QuantizedArray {
    quantizer: PQ<f32>,
    quantized_embeddings: Array2<u8>,
    norms: Option<Array1<f32>>,
}

impl QuantizedArray {
    /// Get the quantizer.
    pub(crate) fn quantizer(&self) -> &PQ<f32> {
        &self.quantizer
    }

    /// Get the quantized embeddings.
    pub(crate) fn embeddings(&self) -> ArrayView2<u8> {
        self.quantized_embeddings.view()
    }

    /// Get the norms.
    pub(crate) fn norms(&self) -> Option<ArrayView1<f32>> {
        self.norms.as_ref().map(|n| n.view())
    }
}

impl Storage for QuantizedArray {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        let mut reconstructed = self
            .quantizer
            .reconstruct_vector(self.quantized_embeddings.row(idx));
        if let Some(ref norms) = self.norms {
            reconstructed *= norms[idx];
        }
        CowArray::from(reconstructed)
    }

    fn shape(&self) -> (usize, usize) {
        (
            self.quantized_embeddings.nrows(),
            self.quantizer.reconstructed_len(),
        )
    }
}

struct PQRead {
    n_embeddings: usize,
    quantizer: PQ<f32>,
    read_norms: bool,
}

/// Memory-mapped quantized embedding matrix.
pub struct MmapQuantizedArray {
    quantizer: PQ<f32>,
    quantized_embeddings: Mmap,
    norms: Option<Array1<f32>>,
}

impl MmapQuantizedArray {
    /// Get the quantizer.
    pub(crate) fn quantizer(&self) -> &PQ<f32> {
        &self.quantizer
    }

    /// Get the quantized embeddings.
    pub(crate) fn embeddings(&self) -> ArrayView2<u8> {
        unsafe { self.quantized_embeddings() }
    }

    /// Get the norms.
    pub(crate) fn norms(&self) -> Option<ArrayView1<f32>> {
        self.norms.as_ref().map(|n| n.view())
    }
}

impl MmapQuantizedArray {
    unsafe fn quantized_embeddings(&self) -> ArrayView2<u8> {
        let n_embeddings = self.shape().0;

        ArrayView2::from_shape_ptr(
            (n_embeddings, self.quantizer.quantized_len()),
            self.quantized_embeddings.as_ptr(),
        )
    }
}

impl Storage for MmapQuantizedArray {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        let quantized = unsafe { self.quantized_embeddings() };

        let mut reconstructed = self.quantizer.reconstruct_vector(quantized.row(idx));
        if let Some(norms) = &self.norms {
            reconstructed *= norms[idx];
        }

        CowArray::from(reconstructed)
    }

    fn shape(&self) -> (usize, usize) {
        (
            self.quantized_embeddings.len() / self.quantizer.quantized_len(),
            self.quantizer.reconstructed_len(),
        )
    }
}

impl PyStorage {
    pub(crate) fn load_quantized(read: &mut BufReader<File>, mmap: bool) -> PyResult<Self> {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::QuantizedArray)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read quantized embedding matrix chunk length\n{}",
                e
            ))
        })?;
        let PQRead {
            n_embeddings,
            quantizer,
            read_norms,
        } = Self::read_product_quantizer(read)?;

        let norms = if read_norms {
            let mut norms_vec = vec![0f32; n_embeddings];
            read.read_f32_into::<LittleEndian>(&mut norms_vec)
                .map_err(|e| exceptions::IOError::py_err(format!("Cannot read norms\n{}", e)))?;
            Some(Array1::from(norms_vec))
        } else {
            None
        };
        if mmap {
            Self::mmap_quantized(read, quantizer, n_embeddings, norms)
        } else {
            Self::read_quantized_chunk(read, quantizer, n_embeddings, norms)
        }
    }

    /// Read a quantized chunk.
    pub(crate) fn read_quantized_chunk<R>(
        read: &mut R,
        quantizer: PQ<f32>,
        n_embeddings: usize,
        norms: Option<Array1<f32>>,
    ) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        let mut quantized_embeddings_vec = vec![0u8; n_embeddings * quantizer.quantized_len()];
        read.read_exact(&mut quantized_embeddings_vec)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot read quantized embeddings\n{}", e))
            })?;
        let quantized_embeddings = Array2::from_shape_vec(
            (n_embeddings, quantizer.quantized_len()),
            quantized_embeddings_vec,
        )
        .map_err(|e| exceptions::ValueError::py_err(e.to_string()))?;
        let array = QuantizedArray {
            quantizer,
            quantized_embeddings,
            norms,
        };
        Ok(PyStorage::new(Rc::new(StorageWrap::QuantizedArray(
            Box::new(array),
        ))))
    }

    pub(crate) fn mmap_quantized(
        read: &mut BufReader<File>,
        quantizer: PQ<f32>,
        n_embeddings: usize,
        norms: Option<Array1<f32>>,
    ) -> PyResult<Self> {
        let matrix_len = n_embeddings * quantizer.quantized_len();
        let map = mmap_array(read, matrix_len)?;

        let quantized = MmapQuantizedArray {
            quantizer,
            quantized_embeddings: map,
            norms,
        };
        Ok(PyStorage::new(Rc::new(StorageWrap::MmapQuantizedArray(
            quantized,
        ))))
    }

    // Helper method to read the product quantizer.
    fn read_product_quantizer<R>(read: &mut R) -> PyResult<PQRead>
    where
        R: Read + Seek,
    {
        let projection = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read quantized embedding matrix projection\n{}",
                e
            ))
        })? != 0;
        let read_norms = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read quantized embedding matrix norms\n{}",
                e
            ))
        })? != 0;
        let quantized_len = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read quantized embedding length\n{}", e))
        })? as usize;
        let reconstructed_len = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read reconstructed embedding length\n{}",
                e
            ))
        })? as usize;
        let n_centroids = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read number of subquantizers\n{}", e))
        })? as usize;
        let n_embeddings = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of quantized embeddings\n{}",
                e
            ))
        })? as usize;

        Self::check_quantizer_invariants(quantized_len, reconstructed_len)?;

        // Quantized storage type.
        u8::ensure_data_type(read).map_err(|e| exceptions::IOError::py_err(e.to_string()))?;

        // Reconstructed embedding type.
        f32::ensure_data_type(read).map_err(|e| exceptions::IOError::py_err(e.to_string()))?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip padding\n{}", e)))?;

        let projection = if projection {
            let mut projection_vec = vec![0f32; reconstructed_len * reconstructed_len];
            read.read_f32_into::<LittleEndian>(&mut projection_vec)
                .map_err(|e| {
                    exceptions::IOError::py_err(format!("Cannot read projection matrix\n{}", e))
                })?;
            Some(
                Array2::from_shape_vec((reconstructed_len, reconstructed_len), projection_vec)
                    .map_err(|e| exceptions::ValueError::py_err(e.to_string()))?,
            )
        } else {
            None
        };

        let quantizer_shape = (
            quantized_len,
            n_centroids,
            reconstructed_len / quantized_len,
        )
            .into_dimension();
        let mut quantizers = vec![0f32; quantizer_shape.size()];
        read.read_f32_into::<LittleEndian>(&mut quantizers)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot read subquantizer\n{}", e)))?;

        Ok(PQRead {
            n_embeddings,
            quantizer: PQ::new(
                projection,
                Array::from_shape_vec(quantizer_shape, quantizers)
                    .expect("Incorrect quantizer shape"),
            ),
            read_norms,
        })
    }

    fn check_quantizer_invariants(quantized_len: usize, reconstructed_len: usize) -> PyResult<()> {
        if reconstructed_len % quantized_len != 0 {
            return Err(exceptions::ValueError::py_err(format!("Reconstructed embedding length ({}) not a multiple of the quantized embedding length: ({})", quantized_len, reconstructed_len)));
        }

        Ok(())
    }

    /// Write a quantized chunk.
    pub(crate) fn write_quantized<W>(
        write: &mut W,
        quantizer: &PQ<f32>,
        quantized: ArrayView2<u8>,
        norms: Option<ArrayView1<f32>>,
    ) -> PyResult<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::QuantizedArray as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write quantized embedding matrix chunk identifier\n{}",
                    e
                ))
            })?;

        // projection (u32), use_norms (u32), quantized_len (u32),
        // reconstructed_len (u32), n_centroids (u32), rows (u64),
        // types (2 x u32 bytes), padding, projection matrix,
        // centroids, norms, quantized data.
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);

        let chunk_size = size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u64>()
            + 2 * size_of::<u32>()
            + n_padding as usize
            + quantizer.projection().is_some() as usize
                * quantizer.reconstructed_len()
                * quantizer.reconstructed_len()
                * size_of::<f32>()
            + quantizer.quantized_len()
                * quantizer.n_quantizer_centroids()
                * (quantizer.reconstructed_len() / quantizer.quantized_len())
                * size_of::<f32>()
            + norms.is_some() as usize * quantized.nrows() * size_of::<f32>()
            + quantized.nrows() * quantizer.quantized_len();

        write
            .write_u64::<LittleEndian>(chunk_size as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write quantized embedding matrix chunk length\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(quantizer.projection().is_some() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write quantized embedding matrix projection\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(norms.is_some() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write quantized embedding matrix norms flag\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(quantizer.quantized_len() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write quantized embedding length\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(quantizer.reconstructed_len() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write reconstructed embedding length\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(quantizer.n_quantizer_centroids() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write number of subquantizers\n{}", e))
            })?;
        write
            .write_u64::<LittleEndian>(quantized.nrows() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write number of quantized embeddings\n{}",
                    e
                ))
            })?;

        // Quantized and reconstruction types.
        write
            .write_u32::<LittleEndian>(u8::type_id())
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write quantized embedding type identifier\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id())
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write reconstructed embedding type identifier\n{}",
                    e
                ))
            })?;

        let padding = vec![0u8; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write padding\n{}", e)))?;

        // Write projection matrix.
        if let Some(projection) = quantizer.projection() {
            for row in projection.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col).map_err(|e| {
                        exceptions::IOError::py_err(format!(
                            "Cannot write projection matrix component\n{}",
                            e
                        ))
                    })?;
                }
            }
        }

        // Write subquantizers.
        for subquantizer in quantizer.subquantizers().outer_iter() {
            for row in subquantizer.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col).map_err(|e| {
                        exceptions::IOError::py_err(format!(
                            "Cannot write subquantizer component\n{}",
                            e
                        ))
                    })?;
                }
            }
        }

        // Write norms.
        if let Some(ref norms) = norms {
            for row in norms.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col).map_err(|e| {
                        exceptions::IOError::py_err(format!(
                            "Cannot write norm vector component\n{}",
                            e
                        ))
                    })?;
                }
            }
        }

        // Write quantized embedding matrix.
        for row in quantized.outer_iter() {
            for &col in row {
                write.write_u8(col).map_err(|e| {
                    exceptions::IOError::py_err(format!(
                        "Cannot write quantized embedding matrix component\n{}",
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }
}

/// Quantizes a viewable storage with the given parameters.
pub(crate) fn quantize_<S>(
    s: &S,
    n_subquantizers: usize,
    n_subquantizer_bits: u32,
    n_iterations: usize,
    n_attempts: usize,
    normalize: bool,
) -> PyResult<QuantizedArray>
where
    S: StorageView,
{
    let (embeds, norms) = if normalize {
        let norms = s.view().outer_iter().map(|e| e.dot(&e).sqrt()).collect();
        let mut normalized = s.view().to_owned();
        for (mut embedding, &norm) in normalized.outer_iter_mut().zip(&norms) {
            embedding /= norm;
        }
        (CowArray::from(normalized), Some(norms))
    } else {
        (CowArray::from(s.view()), None)
    };

    let quantizer = PQ::train_pq_using(
        n_subquantizers,
        n_subquantizer_bits,
        n_iterations,
        n_attempts,
        embeds.view(),
        XorShiftRng::from_entropy(),
    );

    let quantized_embeddings = quantizer.quantize_batch(embeds.view());

    Ok(QuantizedArray {
        quantizer,
        quantized_embeddings,
        norms,
    })
}
