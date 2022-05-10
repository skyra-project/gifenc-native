//! Authors
//! - Kevin Weiner (original Java version - kweiner@fmsware.com)
//! - Thibault Imbert (AS3 version - bytearray.org)
//! - Johan Nordberg (JS version - code@johan-nordberg.com)
//! - Antonio Román (TS version - kyradiscord@gmail.com)
//! - Tyler J. Russell (Rust version - t@k-cs.co)
//!
//! Acknowledgements
//! - GIFCOMPR.C - GIF Image compression routines
//! - Lempel-Ziv compression based on 'compress'. GIF modifications by
//! - David Rowley (mgardi@watdcsu.waterloo.edu)
//!   GIF Image compression - modified 'compress'
//!   Based on: compress.c - File compression ala IEEE Computer, June 1984.
//!   By Authors:
//!   - Spencer W. Thomas (decvax!harpo!utah-cs!utah-gr!thomas)
//!   - Jim McKie (decvax!mcvax!jim)
//!   - Steve Davies (decvax!vax135!petsd!peora!srd)
//!   - Ken Turkowski (decvax!decwrl!turtlevax!ken)
//!   - James A. Woods (decvax!ihnp4!ames!jaw)
//!   - Joe Orost (decvax!vax135!petsd!joe)

use bytes::{BufMut, BytesMut};
use derivative::Derivative;
use napi_derive::napi;

const EOF: i32 = -1;
const BITS: u8 = 12;
const HASH_SIZE: usize = 5003; // 80% occupancy
const MASKS: [u16; 17] = [
	0x0000, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
	0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff,
];

/// @summary
/// Algorithm: use open addressing double hashing (no chaining) on the prefix code / next character combination.
///
/// We do a variant of Knuth's algorithm D (vol. 3, sec. 6.4) along with G. Knott's relatively-prime secondary probe.
/// Here, the modular division first probe is gives way to a faster exclusive-or manipulation. Also do block compression
/// with an adaptive reset, whereby the code table is cleared when the compression ratio decreases, but after the table
/// fills. The variable-length output codes are re-sized at this point, and a special CLEAR code is generated for the
/// decompression.
///
/// **Late addition**: construct the table according to file size for noticeable speed improvement on small files. Please
/// direct questions about this implementation to ames!jaw.
#[napi]
#[derive(Derivative)]
#[derivative(Default)]
pub struct LZWEncoder {
	width: u16,
	height: u16,
	pixels: BytesMut,
	init_code_size: u8,
	current_acc: usize,
	current_bits: usize,
	current_pixel: usize,
	acc: u8,
	first_unused_entry: usize,
	max_code: usize,
	remaining: u32,
	bit_size: usize,

	clear_flag: bool,
	global_initial_bits: usize,
	clear_code: usize,
	end_of_frame_code: usize,

	#[derivative(Default(value = "BytesMut::from([0u8; 256].as_slice())"))]
	accs: BytesMut,
	#[derivative(Default(value = "[0; HASH_SIZE]"))]
	hashes: [i32; HASH_SIZE],
	#[derivative(Default(value = "[0; HASH_SIZE]"))]
	codes: [i32; HASH_SIZE],
}

impl LZWEncoder {
	pub fn new(
		width: u16,
		height: u16,
		pixels: BytesMut,
		color_depth: u8,
	) -> Self {
		LZWEncoder {
			width,
			height,
			pixels,
			init_code_size: u8::max(2, color_depth),
			..Default::default()
		}
	}

	pub fn encode(&mut self, output: BytesMut) {
		output.put_u8(self.init_code_size);
		self.remaining = self.width as u32 * self.height as u32;
		self.current_pixel = 0;
		self.compress(self.init_code_size as usize + 1, output);
		output.put_u8(0);
	}

	fn compress(&mut self, init_bits: usize, output: BytesMut) {
		self.global_initial_bits = init_bits;
		self.clear_flag = false;
		self.bit_size = self.global_initial_bits;
		self.max_code = self.get_max_code(self.bit_size);
		self.clear_code = 1 << (init_bits - 1);
		self.end_of_frame_code = self.clear_code + 1;
		self.first_unused_entry = self.clear_code + 2;
		self.acc = 0;

		let mut code = self.next_pixel();
		let mut hash = 80048;
		let hash_shift = 4;
		self.reset_hash_range(HASH_SIZE);
		self.proc_output(self.clear_code, output);

		let mut c = self.next_pixel();
		'outer: while c != EOF as u8 {
			hash = ((c << BITS) + code) as i32;

			let mut i = ((c << hash_shift) ^ code) as usize;
			if self.hashes[i] == hash {
				code = self.codes[i] as u8;
				continue;
			}

			if self.hashes[i] >= 0 {
				let mut dispose = if i == 0 { 1 } else { HASH_SIZE - i };
				loop {
					i -= dispose;
					if i < 0 {
						i += HASH_SIZE;
					}
					if self.hashes[i] == hash {
						code = self.codes[i] as u8;
						continue 'outer;
					}
					if !self.hashes[i] >= 0 {
						break;
					}
				}
			}

			self.proc_output(code as usize, output);
			code = c;
			if self.first_unused_entry < 1 << BITS {
				self.codes[i] = self.first_unused_entry as i32;
				self.first_unused_entry += 1;
				self.hashes[i] = hash;
			} else {
				self.clear_code_table(output);
			}

			c = self.next_pixel();
		}

		self.proc_output(code as usize, output);
		self.proc_output(self.end_of_frame_code, output);
	}

	fn add_char(&mut self, c: char, output: BytesMut) {
		self.accs[self.acc as usize] = c as u8;
		self.acc += 1;
		if self.acc >= 254 {
			self.flush_packet(output);
		}
	}

	fn clear_code_table(&mut self, output: BytesMut) {
		self.reset_hash_range(HASH_SIZE);
		self.first_unused_entry = self.clear_code + 2;
		self.clear_flag = true;
		self.proc_output(self.clear_code, output);
	}

	#[inline]
	fn reset_hash_range(&mut self, hash_size: usize) {
		self.hashes[..hash_size].fill(-1);
	}

	#[inline]
	fn flush_packet(&mut self, output: BytesMut) {
		if self.acc > 0 {
			output.put_u8(self.acc as u8);
			output[..self.acc as usize]
				.copy_from_slice(&self.accs[..self.acc as usize]);
			self.acc = 0;
		}
	}

	#[inline]
	fn get_max_code(&self, size: usize) -> usize {
		(1 << size) - 1
	}

	fn next_pixel(&mut self) -> u8 {
		if self.remaining == 0 {
			EOF as u8
		} else {
			self.remaining -= 1;
			let pixel = self.pixels[self.current_pixel];
			self.current_pixel += 1;
			pixel & 0xff
		}
	}

	fn proc_output(&mut self, code: usize, output: BytesMut) {
		self.current_acc &= MASKS[self.current_bits] as usize;
		self.current_acc = if self.current_bits > 0 {
			self.current_acc |= code << self.current_bits;
			self.current_acc
		} else {
			code
		};

		self.current_bits += self.bit_size;

		while self.current_bits >= 8 {
			self.add_char((self.current_acc as u8 & 0xff) as char, output);
			self.current_acc >>= 8;
			self.current_bits -= 8;
		}

		if self.first_unused_entry > self.max_code || self.clear_flag {
			if self.clear_flag {
				self.bit_size = self.global_initial_bits;
				self.max_code = self.get_max_code(self.bit_size);
				self.clear_flag = false;
			} else {
				self.bit_size += 1;
				self.max_code = if self.bit_size == BITS as usize {
					1 << BITS
				} else {
					self.get_max_code(self.bit_size)
				};
			}
		}

		if code == self.end_of_frame_code {
			while self.current_bits >= 0 {
				self.add_char((self.current_acc as u8 & 0xff) as char, output);
				self.current_acc >>= 8;
				self.current_bits -= 8;
			}
			self.flush_packet(output);
		}
	}
}
