use bitvec::prelude::*;
use bytes::{BufMut, BytesMut};
use derivative::Derivative;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::{lzw_encoder::LZWEncoder, neuquant::NeuQuant};

const GIF_HEADER: &[u8] = "GIF89a".as_bytes();
const NETSCAPE_HEADER: [u8; 11] =
	[0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30];

// Color table size (bits - 1)
const PALETTE_SIZE: usize = 7;

// TODO: make PALETTE_SIZE and DispoalCode u3?
#[napi]
#[derive(Derivative)]
#[derivative(Default)]
pub enum DisposalCode {
	#[derivative(Default)]
	Unspecified,
	NoDispose,
	RestoreBackground,
	RestorePrevious,
}

impl From<u8> for DisposalCode {
	fn from(n: u8) -> Self {
		match n {
			0 => DisposalCode::Unspecified,
			1 => DisposalCode::NoDispose,
			2 => DisposalCode::RestoreBackground,
			3 => DisposalCode::RestorePrevious,
			_ => unimplemented!(),
		}
	}
}

#[napi]
pub struct EncoderOpts {
	pub delay: i64,
	pub frame_rate: u32,
	pub dispose: DisposalCode,
	pub repeat: i32,
	pub transparent: Option<i32>,
	pub quality: u32,
}

#[napi]
#[derive(Default)]
pub struct GifEncoder {
	pub width: i32,
	pub height: i32,
	transparent: Option<f64>,
	transparent_idx: usize,
	repeat: i32,
	delay: i64,
	image: Option<BytesMut>,
	pixels: Option<BytesMut>,
	idxed_pixels: Option<BytesMut>,
	color_depth: u8,
	color_palette: Option<Vec<f64>>,
	used_entry: BitVec,
	disposal_mode: DisposalCode,
	first_frame: bool,
	sample: u16,
	started: bool,
	readable_streams: Vec<Buffer>,
	byte_buf: BytesMut,
}

#[napi]
impl GifEncoder {
	#[napi(constructor)]
	pub fn new(width: i32, height: i32) -> Self {
		Self { width, height, ..Self::default() }
	}

	#[napi]
	pub fn create_read_stream(&self) -> Buffer {
		unimplemented!()
	}

	pub fn create_write_stream(&self, opts: EncoderOpts) -> Buffer {
		unimplemented!()
	}

	#[napi]
	pub fn set_delay(&mut self, delay: i64) -> &Self {
		self.delay = delay / 10;
		self
	}

	#[napi]
	pub fn set_framerate(&mut self, fps: i64) -> &Self {
		self.delay = 100 / fps;
		self
	}
}
