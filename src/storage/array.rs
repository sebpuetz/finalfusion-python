use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use finalfusion::chunks::storage::NdArray;
use memmap::{Mmap, MmapOptions};
use ndarray::{Array2, ArrayView2, CowArray, Dimension, Ix1, Ix2};
use pyo3::exceptions;
use pyo3::prelude::*;

use crate::io::{padding, ChunkIdentifier, TypeId};
use crate::storage::{PyStorage, StorageWrap};
use finalfusion::prelude::{Storage, StorageView};
use std::fs::File;

impl PyStorage {
    pub(crate) fn read_array_storage<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read embedding matrix chunk length\n{}", e))
        })?;

        let rows = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of rows of the embedding matrix\n{}",
                e
            ))
        })? as usize;
        let cols = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of columns of the embedding matrix\n{}",
                e
            ))
        })? as usize;

        // The components of the embedding matrix should be of type f32.
        f32::ensure_data_type(read).map_err(|e| exceptions::IOError::py_err(e.to_string()))?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip padding\n{}", e)))?;

        let mut data = vec![0f32; rows * cols];
        read.read_f32_into::<LittleEndian>(&mut data).map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read embedding matrix\n{}", e))
        })?;

        let array = Array2::from_shape_vec((rows, cols), data).map_err(|e| {
            exceptions::ValueError::py_err(format!("Invalid storage shape:\n{}", e))
        })?;
        Ok(PyStorage::new(Rc::new(NdArray::new(array).into())))
    }

    pub(crate) fn mmap_array(read: &mut BufReader<File>) -> PyResult<Self> {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read embedding matrix chunk length\n{}", e))
        })?;

        let rows = read.read_u64::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of rows of the embedding matrix\n{}",
                e
            ))
        })? as usize;
        let cols = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot read number of columns of the embedding matrix\n{}",
                e
            ))
        })? as usize;
        let shape = Ix2(rows, cols);

        // The components of the embedding matrix should be of type f32.
        f32::ensure_data_type(read).map_err(|e| exceptions::IOError::py_err(e.to_string()))?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot skip padding\n{}", e)))?;

        // Set up memory mapping.
        let matrix_len = shape.size() * size_of::<f32>();
        let offset = read.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for memory mapping embedding matrix\n{}",
                e,
            ))
        })?;
        let mut mmap_opts = MmapOptions::new();
        let map = unsafe {
            mmap_opts
                .offset(offset)
                .len(matrix_len)
                .map(&read.get_ref())
                .map_err(|e| {
                    exceptions::IOError::py_err(format!(
                        "Cannot memory map embedding matrix\n{}",
                        e
                    ))
                })?
        };
        // Position the reader after the matrix.
        read.seek(SeekFrom::Current(matrix_len as i64))
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot skip embedding matrix\n{}", e))
            })?;
        let mmap = StorageWrap::MmapArray(MmapArray { map, shape });
        Ok(PyStorage::new(Rc::new(mmap)))
    }

    pub(crate) fn write_array_chunk<W>(write: &mut W, data: ArrayView2<f32>) -> PyResult<()>
    where
        W: Seek + Write,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write embedding matrix chunk identifier\n{}",
                    e
                ))
            })?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            exceptions::IOError::py_err(format!(
                "Cannot get file position for computing padding\n{}",
                e
            ))
        })?);

        // Chunk size: rows (u64), columns (u32), type id (u32),
        //             padding ([0,4) bytes), matrix.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + n_padding as usize
            + (data.nrows() * data.ncols() * size_of::<f32>());
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write embedding matrix chunk length\n{}",
                    e
                ))
            })?;
        write
            .write_u64::<LittleEndian>(data.nrows() as u64)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write number of rows of the embedding matrix\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(data.ncols() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write number of columns of the embedding matrix\n{}",
                    e
                ))
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!(
                    "Cannot write embeddings matrix type identifier\n{}",
                    e
                ))
            })?;

        // Write padding, such that the embedding matrix starts on at
        // a multiple of the size of f32 (4 bytes). This is necessary
        // for memory mapping a matrix. Interpreting the raw u8 data
        // as a proper f32 array requires that the data is aligned in
        // memory. However, we cannot always memory map the starting
        // offset of the matrix directly, since mmap(2) requires a
        // file offset that is page-aligned. Since the page size is
        // always a larger power of 2 (e.g. 2^12), which is divisible
        // by 4, the offset of the matrix with regards to the page
        // boundary is also a multiple of 4.

        let padding = vec![0; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write padding\n{}", e)))?;

        for row in data.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col).map_err(|e| {
                    exceptions::IOError::py_err(format!(
                        "Cannot write embedding matrix component\n{}",
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }
}

pub struct MmapArray {
    map: Mmap,
    shape: Ix2,
}

impl Storage for MmapArray {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        CowArray::from(
            // Alignment is ok, padding guarantees that the pointer is at
            // a multiple of 4.
            #[allow(clippy::cast_ptr_alignment)]
            unsafe { ArrayView2::from_shape_ptr(self.shape, self.map.as_ptr() as *const f32) }
                .row(idx)
                .to_owned(),
        )
    }

    fn shape(&self) -> (usize, usize) {
        self.shape.into_pattern()
    }
}

impl StorageView for MmapArray {
    fn view(&self) -> ArrayView2<f32> {
        // Alignment is ok, padding guarantees that the pointer is at
        // a multiple of 4.
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            ArrayView2::from_shape_ptr(self.shape, self.map.as_ptr() as *const f32)
        }
    }
}
