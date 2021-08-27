#![doc(
    html_logo_url = "https://raw.githubusercontent.com/benjaminjellis/oxide/master/docs/static/logo.png?token=AH2AL55YBP3VGCSZZOZI6QTAXU27C"
)]
#![warn(missing_docs)]
//! # oxdie
//! experimental library for simple, high-performance ML in Rust with hardware acceleration
//!
extern crate anyhow;
extern crate arrayfire as af;
extern crate polars;
extern crate rand;

/// Base model trait
pub use self::base::BaseModel;
/// Base
pub mod base;

/// Linear Models
pub mod linear_model;

/// Ensemble Models
pub mod ensemble_model;

/// Tree Models
pub mod tree;

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

/// Find what backend are available and set in order of preference
pub fn set_backend(preference: &str) {
    let available_backends = af::get_available_backends();
    println!("Available backends: {:?}", available_backends);
    match preference {
        "CUDA" => _set_backend_and_confirm(af::Backend::CUDA),
        "CPU" => _set_backend_and_confirm(af::Backend::CPU),
        "OPENCL" => _set_backend_and_confirm(af::Backend::OPENCL),
        _ => panic!("did not recognise backend preference"),
    }
}

/// Helper to set the function and audit it
/// # Parameters
/// - backend: which backend to set
fn _set_backend_and_confirm(backend: af::Backend){
    af::set_backend(backend);
    println!("Using backend: {:?}", backend);
}

/// Set random seed for random array generation
pub fn set_seed(seed: u64) {
    af::set_seed(seed);
}
