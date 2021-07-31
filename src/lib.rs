#![doc(
    html_logo_url = "https://raw.githubusercontent.com/benjaminjellis/oxide/master/docs/static/logo.png?token=AH2AL55YBP3VGCSZZOZI6QTAXU27C"
)]
#![warn(missing_docs)]
//! # oxdie
//! experimental library for simple, high-performance ML in Rust with hardware acceleration
//!
extern crate arrayfire as af;
extern crate polars;
extern crate rand;

/// Base
pub mod base;

/// Linear Models
pub mod linear_model;

/// Activation functions
pub mod activations;

/// General Utils
pub mod utils;

/// Loss functions
pub mod loss;

/// Metrics for training and evaluation
pub mod metrics;

/// Toy Datasets
pub mod datasets;

/// Data Preprocessing
pub mod preprocessing;
