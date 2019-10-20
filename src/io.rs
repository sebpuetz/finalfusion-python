use std::fmt::{self, Display};
use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use itertools::Itertools;
use pyo3::{exceptions, PyResult};

const MODEL_VERSION: u32 = 0;

const MAGIC: [u8; 4] = [b'F', b'i', b'F', b'u'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ChunkIdentifier {
    Header = 0,
    SimpleVocab = 1,
    NdArray = 2,
    BucketSubwordVocab = 3,
    QuantizedArray = 4,
    Metadata = 5,
    NdNorms = 6,
    FastTextSubwordVocab = 7,
    ExplicitSubwordVocab = 8,
}

impl ChunkIdentifier {
    pub fn try_from(identifier: u32) -> Option<Self> {
        use self::ChunkIdentifier::*;

        match identifier {
            1 => Some(SimpleVocab),
            2 => Some(NdArray),
            3 => Some(BucketSubwordVocab),
            4 => Some(QuantizedArray),
            5 => Some(Metadata),
            6 => Some(NdNorms),
            7 => Some(FastTextSubwordVocab),
            8 => Some(ExplicitSubwordVocab),
            _ => None,
        }
    }

    /// Read and ensure that the chunk has the given identifier.
    pub fn ensure_chunk_type<R>(read: &mut R, identifier: ChunkIdentifier) -> PyResult<()>
    where
        R: Read,
    {
        let chunk_id = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read chunk identifier\n{}", e))
        })?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id).ok_or_else(|| {
            exceptions::ValueError::py_err(format!("Unknown chunk identifier: {}", chunk_id))
        })?;
        if chunk_id != identifier {
            return Err(exceptions::ValueError::py_err(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                identifier, chunk_id
            )));
        }

        Ok(())
    }
}

impl Display for ChunkIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::ChunkIdentifier::*;

        match self {
            Header => write!(f, "Header"),
            SimpleVocab => write!(f, "SimpleVocab"),
            NdArray => write!(f, "NdArray"),
            FastTextSubwordVocab => write!(f, "FastTextSubwordVocab"),
            ExplicitSubwordVocab => write!(f, "ExplicitSubwordVocab"),
            BucketSubwordVocab => write!(f, "BucketSubwordVocab"),
            QuantizedArray => write!(f, "QuantizedArray"),
            Metadata => write!(f, "Metadata"),
            NdNorms => write!(f, "NdNorms"),
        }
    }
}

pub trait ReadChunk: Sized {
    fn read_chunk<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek;
}

pub trait WriteChunk {
    fn chunk_identifier(&self) -> ChunkIdentifier;
    fn write_chunk<W>(&self, write: &mut W) -> PyResult<()>
    where
        W: Write;
}

pub(crate) fn find_chunk<R>(
    read: &mut R,
    target_ids: &[ChunkIdentifier],
) -> PyResult<ChunkIdentifier>
where
    R: Read + Seek,
{
    read.seek(SeekFrom::Start(0))?;
    let header = Header::read_chunk(read)?;
    if !header
        .chunk_identifiers
        .iter()
        .cloned()
        .any(|header_id| target_ids.iter().any(|&target_id| target_id == header_id))
    {
        return Err(exceptions::IOError::py_err(format!(
            "Header did not contain [{}]",
            target_ids.iter().map(|id| id.to_string()).join(", ")
        )));
    }
    for &id in header.chunk_identifiers.iter() {
        if let ChunkIdentifier::Header = id {
            return Err(exceptions::IOError::py_err(
                "File contains multiple headers.",
            ));
        }
        if target_ids.iter().any(|&target_id| target_id == id) {
            return Ok(id);
        }
        skip_chunk(read, id)?;
    }
    Err(exceptions::IOError::py_err(format!(
        "File did not contain [{}]",
        target_ids.iter().map(|id| id.to_string()).join(", ")
    )))
}

fn skip_chunk<R>(read: &mut R, skip_id: ChunkIdentifier) -> PyResult<()>
where
    R: Read + Seek,
{
    let identifier = read.read_u32::<LittleEndian>().map_err(|e| {
        exceptions::IOError::py_err(format!("Cannot read chunk identifier.\n{}", e))
    })?;
    println!("{}", ChunkIdentifier::try_from(identifier).unwrap());
    println!("{}", skip_id);
    if identifier != skip_id as u32 {
        return Err(exceptions::IOError::py_err(
            "Chunks are not in header-specified order.",
        ));
    }
    let chunk_size = read.read_u64::<LittleEndian>().map_err(|e| {
        exceptions::IOError::py_err(format!("Cannot read chunk size for {}\n{}", identifier, e))
    })?;
    read.seek(SeekFrom::Current(chunk_size as i64))
        .map_err(|e| {
            exceptions::IOError::py_err(format!("Could not skip the current chunk\n{}", e))
        })?;
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Header {
    chunk_identifiers: Vec<ChunkIdentifier>,
}

impl Header {
    pub fn new(chunk_identifiers: impl Into<Vec<ChunkIdentifier>>) -> Self {
        Header {
            chunk_identifiers: chunk_identifiers.into(),
        }
    }

    pub fn chunk_identifiers(&self) -> &[ChunkIdentifier] {
        &self.chunk_identifiers
    }
}

impl WriteChunk for Header {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::Header
    }

    fn write_chunk<W>(&self, write: &mut W) -> PyResult<()>
    where
        W: Write,
    {
        write
            .write_all(&MAGIC)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot write magic\n{}", e)))?;
        write
            .write_u32::<LittleEndian>(MODEL_VERSION)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write model version\n{}", e))
            })?;
        write
            .write_u32::<LittleEndian>(self.chunk_identifiers.len() as u32)
            .map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot write chunk identifiers length\n{}", e))
            })?;

        for &identifier in &self.chunk_identifiers {
            write
                .write_u32::<LittleEndian>(identifier as u32)
                .map_err(|e| {
                    exceptions::IOError::py_err(format!("Cannot write chunk identifier\n{}", e))
                })?;
        }

        Ok(())
    }
}

impl ReadChunk for Header {
    fn read_chunk<R>(read: &mut R) -> PyResult<Self>
    where
        R: Read + Seek,
    {
        // Magic and version ceremony.
        let mut magic = [0u8; 4];
        read.read_exact(&mut magic)
            .map_err(|e| exceptions::IOError::py_err(format!("Cannot read magic\n{}", e)))?;

        if magic != MAGIC {
            return Err(exceptions::ValueError::py_err(format!(
                "Expected 'FiFu' as magic, got: {}",
                String::from_utf8_lossy(&magic).into_owned()
            )));
        }

        let version = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read model version\n{}", e))
        })?;
        if version != MODEL_VERSION {
            return Err(exceptions::ValueError::py_err(format!(
                "Unknown finalfusion version: {}",
                version
            )));
        }

        // Read chunk identifiers.
        let chunk_identifiers_len = read.read_u32::<LittleEndian>().map_err(|e| {
            exceptions::IOError::py_err(format!("Cannot read chunk identifiers length\n{}", e))
        })? as usize;
        let mut chunk_identifiers = Vec::with_capacity(chunk_identifiers_len);
        for _ in 0..chunk_identifiers_len {
            let identifier = read.read_u32::<LittleEndian>().map_err(|e| {
                exceptions::IOError::py_err(format!("Cannot read chunk identifier\n{}", e))
            })?;
            let chunk_identifier = ChunkIdentifier::try_from(identifier).ok_or_else(|| {
                exceptions::ValueError::py_err(format!("Unknown chunk identifier: {}", identifier))
            })?;
            chunk_identifiers.push(chunk_identifier);
        }

        Ok(Header { chunk_identifiers })
    }
}
