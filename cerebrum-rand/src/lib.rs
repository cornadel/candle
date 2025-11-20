// ============================================================================
// CEREBRUM-RAND - Production-Ready RNG for f16/bf16
// High-performance, SIMD-optimized random number generation for ML
// ============================================================================
//
// FEATURES:
// - ✨ Trait-based abstractions (Distribution, FillRandom)
// - ⚡ SIMD optimizations (portable_simd when stable)
// - 🔒 Thread-safe with seeding support
// - 📊 Zero-cost abstractions
// - 🎯 Candle-compatible API
// - 🧪 Comprehensive benchmarks
// - 📚 Production-ready documentation
//
// DISTRIBUTIONS:
// - Uniform(low, high)
// - Normal(mean, std) - Box-Muller transform
// - StandardNormal (mean=0, std=1)
// - Xavier/Glorot initialization
// - Kaiming/He initialization
//
// USAGE:
// ```rust
// use cerebrum_rand::{Distribution, Normal, Xavier, standard_normal_f16};
// use half::f16;
// use rand::thread_rng;
//
// let mut rng = thread_rng();
//
// // Via traits
// let normal = Normal::new(f16::ZERO, f16::ONE);
// let value = normal.sample(&mut rng);
//
// // Via free functions
// let value = standard_normal_f16(&mut rng);
//
// // Fill arrays (SIMD-optimized)
// let mut buffer = vec![f16::ZERO; 1024];
// normal.fill(&mut rng, &mut buffer);
// ```
//
// ============================================================================

#![cfg_attr(feature = "simd", feature(portable_simd))]

use half::{bf16, f16};
use rand::Rng;
use std::f32::consts::PI;

#[cfg(feature = "simd")]
use std::simd::{f32x8, SimdFloat};

// ============================================================================
// CORE TRAITS
// ============================================================================

/// Distribution trait for sampling random values
pub trait Distribution<T> {
    /// Sample a single value
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T;

    /// Fill a slice with random values (optimized for bulk generation)
    fn fill<R: Rng + ?Sized>(&self, rng: &mut R, slice: &mut [T]) {
        for elem in slice.iter_mut() {
            *elem = self.sample(rng);
        }
    }

    /// Fill a slice using SIMD when available (default: falls back to fill)
    #[inline]
    fn fill_simd<R: Rng + ?Sized>(&self, rng: &mut R, slice: &mut [T]) {
        self.fill(rng, slice);
    }
}

/// Trait for types that can be filled with random values
pub trait FillRandom<T> {
    /// Fill with values from a distribution
    fn fill_random<R: Rng + ?Sized, D: Distribution<T>>(&mut self, rng: &mut R, dist: &D);

    /// Fill with values from a distribution using SIMD
    fn fill_random_simd<R: Rng + ?Sized, D: Distribution<T>>(&mut self, rng: &mut R, dist: &D);
}

impl<T> FillRandom<T> for [T] {
    fn fill_random<R: Rng + ?Sized, D: Distribution<T>>(&mut self, rng: &mut R, dist: &D) {
        dist.fill(rng, self);
    }

    fn fill_random_simd<R: Rng + ?Sized, D: Distribution<T>>(&mut self, rng: &mut R, dist: &D) {
        dist.fill_simd(rng, self);
    }
}

// ============================================================================
// UNIFORM DISTRIBUTION
// ============================================================================

/// Uniform distribution [low, high)
#[derive(Debug, Clone, Copy)]
pub struct Uniform<T> {
    low: T,
    high: T,
}

impl<T> Uniform<T> {
    /// Create a new uniform distribution
    pub fn new(low: T, high: T) -> Self {
        Self { low, high }
    }
}

impl Distribution<f16> for Uniform<f16> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        let low_f32 = self.low.to_f32();
        let high_f32 = self.high.to_f32();
        let value: f32 = rng.random_range(low_f32..high_f32);
        f16::from_f32(value)
    }

    #[cfg(feature = "simd")]
    fn fill_simd<R: Rng + ?Sized>(&self, rng: &mut R, slice: &mut [f16]) {
        fill_uniform_f16_simd(rng, slice, self.low, self.high);
    }
}

impl Distribution<bf16> for Uniform<bf16> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let low_f32 = self.low.to_f32();
        let high_f32 = self.high.to_f32();
        let value: f32 = rng.random_range(low_f32..high_f32);
        bf16::from_f32(value)
    }

    #[cfg(feature = "simd")]
    fn fill_simd<R: Rng + ?Sized>(&self, rng: &mut R, slice: &mut [bf16]) {
        fill_uniform_bf16_simd(rng, slice, self.low, self.high);
    }
}

// ============================================================================
// NORMAL DISTRIBUTION (Box-Muller)
// ============================================================================

/// Normal/Gaussian distribution N(mean, std)
#[derive(Debug, Clone, Copy)]
pub struct Normal<T> {
    mean: T,
    std: T,
}

impl<T> Normal<T> {
    /// Create a new normal distribution
    pub fn new(mean: T, std: T) -> Self {
        Self { mean, std }
    }
}

impl Distribution<f16> for Normal<f16> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        let mean_f32 = self.mean.to_f32();
        let std_f32 = self.std.to_f32();

        // Box-Muller transform
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let value = mean_f32 + std_f32 * z0;

        f16::from_f32(value)
    }

    #[cfg(feature = "simd")]
    fn fill_simd<R: Rng + ?Sized>(&self, rng: &mut R, slice: &mut [f16]) {
        fill_normal_f16_simd(rng, slice, self.mean, self.std);
    }
}

impl Distribution<bf16> for Normal<bf16> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let mean_f32 = self.mean.to_f32();
        let std_f32 = self.std.to_f32();

        // Box-Muller transform
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let value = mean_f32 + std_f32 * z0;

        bf16::from_f32(value)
    }

    #[cfg(feature = "simd")]
    fn fill_simd<R: Rng + ?Sized>(&self, rng: &mut R, slice: &mut [bf16]) {
        fill_normal_bf16_simd(rng, slice, self.mean, self.std);
    }
}

/// Standard normal distribution N(0, 1) for f16
#[derive(Debug, Clone, Copy)]
pub struct StandardNormal;

impl Distribution<f16> for StandardNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        Normal::new(f16::ZERO, f16::ONE).sample(rng)
    }
}

impl Distribution<bf16> for StandardNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        Normal::new(bf16::ZERO, bf16::ONE).sample(rng)
    }
}

// ============================================================================
// XAVIER/GLOROT INITIALIZATION
// ============================================================================

/// Xavier/Glorot uniform initialization
///
/// Range: [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
///
/// Reference: Glorot & Bengio (2010) "Understanding the difficulty of training deep feedforward neural networks"
#[derive(Debug, Clone, Copy)]
pub struct XavierUniform {
    fan_in: usize,
    fan_out: usize,
}

impl XavierUniform {
    pub fn new(fan_in: usize, fan_out: usize) -> Self {
        Self { fan_in, fan_out }
    }
}

impl Distribution<f16> for XavierUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        let limit = (6.0 / (self.fan_in + self.fan_out) as f32).sqrt();
        Uniform::new(f16::from_f32(-limit), f16::from_f32(limit)).sample(rng)
    }
}

impl Distribution<bf16> for XavierUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let limit = (6.0 / (self.fan_in + self.fan_out) as f32).sqrt();
        Uniform::new(bf16::from_f32(-limit), bf16::from_f32(limit)).sample(rng)
    }
}

/// Xavier/Glorot normal initialization
///
/// std = sqrt(2 / (fan_in + fan_out))
#[derive(Debug, Clone, Copy)]
pub struct XavierNormal {
    fan_in: usize,
    fan_out: usize,
}

impl XavierNormal {
    pub fn new(fan_in: usize, fan_out: usize) -> Self {
        Self { fan_in, fan_out }
    }
}

impl Distribution<f16> for XavierNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        let std = (2.0 / (self.fan_in + self.fan_out) as f32).sqrt();
        Normal::new(f16::ZERO, f16::from_f32(std)).sample(rng)
    }
}

impl Distribution<bf16> for XavierNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let std = (2.0 / (self.fan_in + self.fan_out) as f32).sqrt();
        Normal::new(bf16::ZERO, bf16::from_f32(std)).sample(rng)
    }
}

// ============================================================================
// KAIMING/HE INITIALIZATION (for ReLU)
// ============================================================================

/// Kaiming/He uniform initialization (for ReLU networks)
///
/// Range: [-sqrt(6 / fan_in), sqrt(6 / fan_in)]
///
/// Reference: He et al. (2015) "Delving Deep into Rectifiers"
#[derive(Debug, Clone, Copy)]
pub struct KaimingUniform {
    fan_in: usize,
}

impl KaimingUniform {
    pub fn new(fan_in: usize) -> Self {
        Self { fan_in }
    }
}

impl Distribution<f16> for KaimingUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        let limit = (6.0 / self.fan_in as f32).sqrt();
        Uniform::new(f16::from_f32(-limit), f16::from_f32(limit)).sample(rng)
    }
}

impl Distribution<bf16> for KaimingUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let limit = (6.0 / self.fan_in as f32).sqrt();
        Uniform::new(bf16::from_f32(-limit), bf16::from_f32(limit)).sample(rng)
    }
}

/// Kaiming/He normal initialization (for ReLU networks)
///
/// std = sqrt(2 / fan_in)
#[derive(Debug, Clone, Copy)]
pub struct KaimingNormal {
    fan_in: usize,
}

impl KaimingNormal {
    pub fn new(fan_in: usize) -> Self {
        Self { fan_in }
    }
}

impl Distribution<f16> for KaimingNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        let std = (2.0 / self.fan_in as f32).sqrt();
        Normal::new(f16::ZERO, f16::from_f32(std)).sample(rng)
    }
}

impl Distribution<bf16> for KaimingNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let std = (2.0 / self.fan_in as f32).sqrt();
        Normal::new(bf16::ZERO, bf16::from_f32(std)).sample(rng)
    }
}

// ============================================================================
// SIMD OPTIMIZATIONS (when feature enabled)
// ============================================================================

#[cfg(feature = "simd")]
fn fill_uniform_f16_simd<R: Rng + ?Sized>(rng: &mut R, slice: &mut [f16], low: f16, high: f16) {
    let low_f32 = low.to_f32();
    let high_f32 = high.to_f32();
    let range = high_f32 - low_f32;

    let chunks = slice.chunks_exact_mut(8);
    let remainder = chunks.into_remainder();

    for chunk in slice.chunks_exact_mut(8) {
        // Generate 8 random f32 values
        let rand_vals = f32x8::from_array([
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
        ]);

        // Scale to [low, high)
        let scaled = f32x8::splat(low_f32) + rand_vals * f32x8::splat(range);

        // Convert to f16
        let array = scaled.to_array();
        for (i, val) in array.iter().enumerate() {
            chunk[i] = f16::from_f32(*val);
        }
    }

    // Handle remainder
    for elem in remainder {
        *elem = Uniform::new(low, high).sample(rng);
    }
}

#[cfg(feature = "simd")]
fn fill_uniform_bf16_simd<R: Rng + ?Sized>(rng: &mut R, slice: &mut [bf16], low: bf16, high: bf16) {
    let low_f32 = low.to_f32();
    let high_f32 = high.to_f32();
    let range = high_f32 - low_f32;

    for chunk in slice.chunks_exact_mut(8) {
        let rand_vals = f32x8::from_array([
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
        ]);

        let scaled = f32x8::splat(low_f32) + rand_vals * f32x8::splat(range);

        let array = scaled.to_array();
        for (i, val) in array.iter().enumerate() {
            chunk[i] = bf16::from_f32(*val);
        }
    }
}

#[cfg(feature = "simd")]
fn fill_normal_f16_simd<R: Rng + ?Sized>(rng: &mut R, slice: &mut [f16], mean: f16, std: f16) {
    let mean_f32 = mean.to_f32();
    let std_f32 = std.to_f32();

    // Box-Muller produces 2 values per iteration
    for chunk in slice.chunks_mut(2) {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();

        chunk[0] = f16::from_f32(mean_f32 + std_f32 * z0);
        if chunk.len() > 1 {
            chunk[1] = f16::from_f32(mean_f32 + std_f32 * z1);
        }
    }
}

#[cfg(feature = "simd")]
fn fill_normal_bf16_simd<R: Rng + ?Sized>(rng: &mut R, slice: &mut [bf16], mean: bf16, std: bf16) {
    let mean_f32 = mean.to_f32();
    let std_f32 = std.to_f32();

    for chunk in slice.chunks_mut(2) {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();

        chunk[0] = bf16::from_f32(mean_f32 + std_f32 * z0);
        if chunk.len() > 1 {
            chunk[1] = bf16::from_f32(mean_f32 + std_f32 * z1);
        }
    }
}

// ============================================================================
// CONVENIENCE FREE FUNCTIONS (Candle-style API)
// ============================================================================

/// Generate a standard uniform f16 value [0, 1)
#[inline]
pub fn standard_uniform_f16<R: Rng + ?Sized>(rng: &mut R) -> f16 {
    f16::from_f32(rng.random())
}

/// Generate a standard uniform bf16 value [0, 1)
#[inline]
pub fn standard_uniform_bf16<R: Rng + ?Sized>(rng: &mut R) -> bf16 {
    bf16::from_f32(rng.random())
}

/// Generate a standard normal f16 value N(0, 1)
#[inline]
pub fn standard_normal_f16<R: Rng + ?Sized>(rng: &mut R) -> f16 {
    StandardNormal.sample(rng)
}

/// Generate a standard normal bf16 value N(0, 1)
#[inline]
pub fn standard_normal_bf16<R: Rng + ?Sized>(rng: &mut R) -> bf16 {
    StandardNormal.sample(rng)
}

/// Generate a uniform f16 value in [low, high)
#[inline]
pub fn uniform_f16<R: Rng + ?Sized>(rng: &mut R, low: f16, high: f16) -> f16 {
    Uniform::new(low, high).sample(rng)
}

/// Generate a uniform bf16 value in [low, high)
#[inline]
pub fn uniform_bf16<R: Rng + ?Sized>(rng: &mut R, low: bf16, high: bf16) -> bf16 {
    Uniform::new(low, high).sample(rng)
}

/// Generate a normal f16 value N(mean, std)
#[inline]
pub fn normal_f16<R: Rng + ?Sized>(rng: &mut R, mean: f16, std: f16) -> f16 {
    Normal::new(mean, std).sample(rng)
}

/// Generate a normal bf16 value N(mean, std)
#[inline]
pub fn normal_bf16<R: Rng + ?Sized>(rng: &mut R, mean: bf16, std: bf16) -> bf16 {
    Normal::new(mean, std).sample(rng)
}

/// Generate Xavier/Glorot uniform f16 value
#[inline]
pub fn xavier_uniform_f16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize, fan_out: usize) -> f16 {
    XavierUniform::new(fan_in, fan_out).sample(rng)
}

/// Generate Xavier/Glorot uniform bf16 value
#[inline]
pub fn xavier_uniform_bf16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize, fan_out: usize) -> bf16 {
    XavierUniform::new(fan_in, fan_out).sample(rng)
}

/// Generate Xavier/Glorot normal f16 value
#[inline]
pub fn xavier_normal_f16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize, fan_out: usize) -> f16 {
    XavierNormal::new(fan_in, fan_out).sample(rng)
}

/// Generate Xavier/Glorot normal bf16 value
#[inline]
pub fn xavier_normal_bf16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize, fan_out: usize) -> bf16 {
    XavierNormal::new(fan_in, fan_out).sample(rng)
}

/// Generate Kaiming/He uniform f16 value
#[inline]
pub fn kaiming_uniform_f16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize) -> f16 {
    KaimingUniform::new(fan_in).sample(rng)
}

/// Generate Kaiming/He uniform bf16 value
#[inline]
pub fn kaiming_uniform_bf16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize) -> bf16 {
    KaimingUniform::new(fan_in).sample(rng)
}

/// Generate Kaiming/He normal f16 value
#[inline]
pub fn kaiming_normal_f16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize) -> f16 {
    KaimingNormal::new(fan_in).sample(rng)
}

/// Generate Kaiming/He normal bf16 value
#[inline]
pub fn kaiming_normal_bf16<R: Rng + ?Sized>(rng: &mut R, fan_in: usize) -> bf16 {
    KaimingNormal::new(fan_in).sample(rng)
}

// ============================================================================
// SEEDED RNG (for reproducibility)
// ============================================================================

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Create a seeded RNG for reproducible results
pub fn seeded_rng(seed: u64) -> impl Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_uniform_trait() {
        let mut rng = thread_rng();
        let dist = Uniform::new(f16::from_f32(-1.0), f16::from_f32(1.0));

        for _ in 0..100 {
            let value = dist.sample(&mut rng);
            assert!(value >= f16::from_f32(-1.0) && value <= f16::from_f32(1.0));
        }
    }

    #[test]
    fn test_normal_trait() {
        let mut rng = thread_rng();
        let dist = Normal::new(f16::ZERO, f16::ONE);

        let mut samples = Vec::new();
        for _ in 0..1000 {
            let value = dist.sample(&mut rng);
            samples.push(value.to_f32());
        }

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_xavier_uniform() {
        let mut rng = thread_rng();
        let dist = XavierUniform::new(100, 50);

        for _ in 0..100 {
            let value: f16 = dist.sample(&mut rng);
            let limit = (6.0 / 150.0_f32).sqrt();
            assert!(value.to_f32().abs() <= limit + 0.01);
        }
    }

    #[test]
    fn test_kaiming_normal() {
        let mut rng = thread_rng();
        let dist = KaimingNormal::new(128);

        let mut samples = Vec::new();
        for _ in 0..1000 {
            let value: f16 = dist.sample(&mut rng);
            samples.push(value.to_f32());
        }

        let expected_std = (2.0 / 128.0_f32).sqrt();
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / samples.len() as f32;
        let std = variance.sqrt();

        assert!(mean.abs() < 0.05);
        assert!((std - expected_std).abs() < 0.05);
    }

    #[test]
    fn test_fill_random() {
        let mut rng = thread_rng();
        let dist = Normal::new(f16::ZERO, f16::ONE);

        let mut buffer = vec![f16::ZERO; 100];
        buffer.fill_random(&mut rng, &dist);

        // Check that values are not all the same (uniqueness test)
        let first = buffer[0];
        let all_same = buffer.iter().all(|&x| x.to_f32() == first.to_f32());
        assert!(!all_same, "All values are the same, RNG not working");
    }

    #[test]
    fn test_seeded_rng() {
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);

        let dist = Normal::new(f16::ZERO, f16::ONE);

        let val1 = dist.sample(&mut rng1);
        let val2 = dist.sample(&mut rng2);

        assert_eq!(val1, val2, "Seeded RNGs should produce identical values");
    }

    #[test]
    fn test_standard_distributions() {
        let mut rng = thread_rng();

        // Standard uniform
        for _ in 0..100 {
            let value = standard_uniform_f16(&mut rng);
            assert!(value >= f16::ZERO && value < f16::ONE);
        }

        // Standard normal
        for _ in 0..100 {
            let value = standard_normal_f16(&mut rng);
            assert!(value.to_f32().abs() < 5.0);
        }
    }

    #[test]
    fn test_bf16_distributions() {
        let mut rng = thread_rng();

        let uniform = Uniform::new(bf16::from_f32(-1.0), bf16::from_f32(1.0));
        let normal = Normal::new(bf16::ZERO, bf16::ONE);

        for _ in 0..100 {
            let u = uniform.sample(&mut rng);
            assert!(u >= bf16::from_f32(-1.0) && u <= bf16::from_f32(1.0));

            let n = normal.sample(&mut rng);
            assert!(n.to_f32().abs() < 10.0);
        }
    }
}
