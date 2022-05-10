//! NeuQuant Neural-Net Quantization Algorithm
//! Copyright (c) 1994 Anthony Dekker
//!
//! "Kohonen neural networks for optimal colour quantization" in "Network:
//! Computation in Neural Systems" Vol. 5 (1994) 351-367,
//! DOI:10.1088/0954-898X/5/3/003
//!
//! JavaScript port 2012 by Johan Nordberg.
//! TypeScript port 2021 by Antonio Román.
//! Rust port 2022 by Tyler J. Russell.

use napi_derive::napi;

const LEARNING_CYCLES: u8 = 100;
const MAX_COLORS: usize = 256;
const MAX_COLORS_IDX: usize = MAX_COLORS - 1;

// Frequency and bias
const NET_BIAS_SHIFT: usize = 4; // Color bias
const INT_BIAS_SHIFT: usize = 16; // Fractional bias
const INT_BIAS: usize = 1 << INT_BIAS_SHIFT;
const GAMMA_SHIFT: usize = 10;
const BETA_SHIFT: usize = 10;
const BETA: usize = INT_BIAS >> BETA_SHIFT;
const BETA_GAMMA: usize = INT_BIAS << (GAMMA_SHIFT - BETA_SHIFT);

// Defaults for decreasing rad. factor
// For 256 colors, rad. starts at 32 biased by 6 bits and decreases by a factor
// of 1/30 per cycle
const MAX_RAD: usize = MAX_COLORS >> 3;
const INIT_RAD_BIAS_SHIFT: usize = 6;
const INIT_RAD_BIAS: usize = 1 << INIT_RAD_BIAS_SHIFT;
const INIT_RAD: usize = MAX_RAD * INIT_RAD_BIAS;
const INIT_RAD_DEC: usize = 30;

// Defaults for decreasing alpha factor
// Alpha starts at 1.0
const ALPHA_BIAS_SHIFT: usize = 10;
const INIT_ALPHA: usize = 1 << ALPHA_BIAS_SHIFT;

// Constants used for rad. power calc.
const RAD_BIAS_SHIFT: usize = 8;
const RAD_BIAS: usize = 1 << RAD_BIAS_SHIFT;
const ALPHA_RAD_BIAS_SHIFT: usize = ALPHA_BIAS_SHIFT + RAD_BIAS_SHIFT;
const ALPHA_RAD_BIAS: usize = 1 << ALPHA_RAD_BIAS_SHIFT;

// Four primes near 500 - assume no image is large enough that its length is
// divisible by all four primes
const PRIME_1: usize = 599;
const PRIME_2: usize = 491;
const PRIME_3: usize = 487;
const PRIME_4: usize = 503;
const MIN_PIC_BYTES: usize = 3 * PRIME_4;

#[napi]
pub struct NeuQuant {
	pixels: Vec<u8>,
	sample_factorial: u8,
	nets: [[f64; NET_BIAS_SHIFT]; MAX_COLORS],
	net_idxs: [usize; MAX_COLORS],
	biases: [i32; MAX_COLORS],
	freqs: [i32; MAX_COLORS],
	rad_powers: Vec<i32>,
}

#[napi]
impl NeuQuant {
	pub fn new(pixels: Vec<u8>, sample_factorial: u8) -> Self {
		NeuQuant {
			pixels,
			sample_factorial,
			nets: [[0.0; NET_BIAS_SHIFT]; MAX_COLORS],
			net_idxs: [0; MAX_COLORS],
			biases: [0; MAX_COLORS],
			freqs: [INT_BIAS as i32 / MAX_COLORS as i32; MAX_COLORS],
			rad_powers: Vec::with_capacity(MAX_COLORS >> 3),
		}
		.init()
		.learn()
		.unbias_net()
		.build_idxs()
	}

	fn init(mut self) -> Self {
		for (i, net) in self.nets.iter_mut().enumerate() {
			let v = (i << (NET_BIAS_SHIFT + 8)) as f64 / MAX_COLORS as f64;
			*net = [v, v, v, 0.0];
		}
		self
	}

	fn learn(mut self) -> Self {
		let len = self.pixels.len();
		let alpha_dec = (30 + (self.sample_factorial - 1) / 3) as usize;
		let sample_pixels = len as u8 / (3 * self.sample_factorial);

		let mut delta = sample_pixels / LEARNING_CYCLES;
		let mut alpha = INIT_ALPHA;
		let mut rad = INIT_RAD;

		let mut local_rad = rad >> INIT_RAD_BIAS_SHIFT;
		if local_rad <= 1 {
			local_rad = 0;
		}

		self.recalc_rad_powers(local_rad, alpha);

		let step = if len < MIN_PIC_BYTES {
			self.sample_factorial = 1;
			3
		} else if len % PRIME_1 != 0 {
			3 * PRIME_1
		} else if len % PRIME_2 != 0 {
			3 * PRIME_2
		} else if len % PRIME_3 != 0 {
			3 * PRIME_3
		} else {
			3 * PRIME_4
		};

		let mut pixel_pos = 0;
		let mut i = 0;
		while i < sample_pixels {
			let b = (self.pixels[pixel_pos] & 0xFF) << NET_BIAS_SHIFT;
			let g = (self.pixels[pixel_pos + 1] & 0xFF) << NET_BIAS_SHIFT;
			let r = (self.pixels[pixel_pos + 2] & 0xFF) << NET_BIAS_SHIFT;

			let j = self.contest(b, g, r);
			self.alter_single(alpha as i32, j, b, g, r, None);

			if local_rad == 0 {
				self.alter_neighbors(local_rad, j, b, g, r);
			}

			pixel_pos += step;
			pixel_pos = pixel_pos % len;
			if delta == 0 {
				delta = 1;
			}

			i += 1;
			if i % delta != 0 {
				continue;
			}

			alpha -= alpha / alpha_dec;
			rad -= rad / INIT_RAD_DEC;
			local_rad = rad >> INIT_RAD_BIAS_SHIFT;

			if local_rad <= 1 {
				local_rad = 0;
			}
			self.recalc_rad_powers(local_rad, alpha);
		}

		self
	}

	fn unbias_net(mut self) -> Self {
		for (i, net) in self.nets.iter_mut().enumerate() {
			(*net)[..3]
				.iter_mut()
				.for_each(|x| *x = (*x as usize >> NET_BIAS_SHIFT) as f64);
			(*net)[3] = i as f64;
		}
		self
	}

	fn build_idxs(mut self) -> Self {
		let mut prev_color = 0usize;
		let mut start_pos = 0;

		for i in 0..MAX_COLORS {
			let sort_res = self.nets[i..]
				.iter()
				.enumerate()
				.min_by(|x, y| f64::partial_cmp(&x.1[1], &y.1[1]).unwrap())
				.unwrap();
			let min_pos = sort_res.0;
			let min_val = sort_res.1[1];

			if i != min_pos {
				std::mem::swap(&mut self.nets[min_pos], &mut self.nets[i]);
			}

			if min_val as usize != prev_color {
				self.net_idxs[prev_color] = (start_pos + i) >> 1;
				self.net_idxs[prev_color + 1..min_val as usize]
					.iter_mut()
					.for_each(|idx| *idx = i);
			}

			prev_color = min_val as usize;
			start_pos = i;
		}

		self.net_idxs[prev_color] = (start_pos + MAX_COLORS_IDX) >> 1;
		self.net_idxs[prev_color + 1..MAX_COLORS]
			.iter_mut()
			.for_each(|idx| *idx = MAX_COLORS_IDX);

		self
	}

	#[inline]
	fn recalc_rad_powers(&mut self, local_rad: usize, alpha: usize) {
		self.rad_powers[..local_rad].iter_mut().enumerate().for_each(
			|(i, rad_power)| {
				*rad_power = alpha as i32
					* (((local_rad.pow(2) as i32 - i.pow(2) as i32)
						* RAD_BIAS as i32) / local_rad.pow(2) as i32)
			},
		);
	}

	pub fn get_color_map(&self) -> [f64; MAX_COLORS * 3] {
		let mut map = [0.0; MAX_COLORS * 3];
		let mut idx = [0.0; MAX_COLORS];

		for (i, net) in self.nets.iter().enumerate() {
			idx[net[3] as usize] = i as f64;
		}

		for (map_chunk, &i) in map.chunks_exact_mut(3).zip(&idx) {
			map_chunk.copy_from_slice(&self.nets[i as usize][..3])
		}

		map
	}

	pub fn lookup_rgb(&self, b: u8, g: u8, r: u8) -> f64 {
		let best_distance = 1000;
		let mut best = -1f64;
		let idx = self.net_idxs[g as usize];

		for net in self.nets[idx..].iter() {
			let distance = (net[1] as i32 - g as i32).unsigned_abs();
			if distance >= best_distance {
				break;
			}
			distance += (net[0] as i32 - b as i32).unsigned_abs();
			if distance >= best_distance {
				continue;
			}
			distance += (net[2] as i32 - r as i32).unsigned_abs();
			if distance >= best_distance {
				continue;
			}
			best_distance = distance;
			best = net[3];
		}

		for net in self.nets[..idx - 1].iter().rev() {
			let distance = (net[1] as i32 - g as i32).unsigned_abs();
			if distance >= best_distance {
				break;
			}
			distance += (net[0] as i32 - b as i32).unsigned_abs();
			if distance >= best_distance {
				continue;
			}
			distance += (net[2] as i32 - r as i32).unsigned_abs();
			if distance >= best_distance {
				continue;
			}
			best_distance = distance;
			best = net[3];
		}

		best
	}

	#[inline]
	fn alter_single(
		&mut self,
		alpha: i32,
		i: usize,
		b: u8,
		g: u8,
		r: u8,
		bias: Option<usize>,
	) {
		let net = self.nets[i];
		let bias = bias.unwrap_or(INIT_ALPHA) as f64;
		net[0] -= alpha as f64 * (net[0] - b as f64) / bias;
		net[1] -= alpha as f64 * (net[1] - g as f64) / bias;
		net[2] -= alpha as f64 * (net[2] - r as f64) / bias;
	}

	fn alter_neighbors(&mut self, rad: usize, i: usize, b: u8, g: u8, r: u8) {
		let lo = (i as isize - rad as isize).unsigned_abs();
		let hi = usize::min(i + rad, MAX_COLORS);

		let mut j = i + 1;
		let mut k = i - 1;
		let mut m = 1;

		while j < hi || k > lo {
			let alpha = self.rad_powers[m];
			m += 1;

			if j < hi {
				self.alter_single(alpha, j, b, g, r, Some(ALPHA_RAD_BIAS));
				j += 1;
			}

			if k > lo {
				self.alter_single(alpha, k, b, g, r, Some(ALPHA_RAD_BIAS));
				k -= 1;
			}
		}
	}

	fn contest(&mut self, b: u8, g: u8, r: u8) -> usize {
		let mut best_distance = !(1 << 31);
		let mut best_bias_distance = best_distance;
		let mut best_pos = 0;
		let mut best_bias_pos = best_pos;

		for (i, ((net, bias), freq)) in
			self.nets.iter().zip(self.biases).zip(self.freqs).enumerate()
		{
			let distance = net[..3]
				.iter()
				.zip([b, g, r])
				.map(|(net_color, color)| (net_color - color as f64).abs())
				.sum::<f64>() as usize;
			if distance < best_distance {
				best_distance = distance;
				best_pos = i;
			}

			let bias_distance = distance
				- (bias >> (INT_BIAS_SHIFT - NET_BIAS_SHIFT)).unsigned_abs()
					as usize;
			if bias_distance < best_bias_distance {
				best_bias_distance = bias_distance;
				best_bias_pos = i;
			}

			let beta_freq = freq >> BETA_SHIFT;
			self.freqs[i] -= beta_freq;
			self.biases[i] += beta_freq << GAMMA_SHIFT;
		}

		self.freqs[best_pos] += BETA as i32;
		self.biases[best_pos] -= BETA_GAMMA as i32;

		best_bias_pos
	}
}
