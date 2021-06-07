#![doc(
    html_logo_url = "https://raw.githubusercontent.com/benjaminjellis/oxide/master/docs/static/logo.png?token=AH2AL55YBP3VGCSZZOZI6QTAXU27C",
)]
#![warn(missing_docs)]

extern crate arrayfire as af;

/// Base for all models
pub mod base;

/// Linear Models
pub mod linear_model;

/// Activation functions
pub mod activations;

/// General Utils
pub mod utils;

/// Loss functions
pub mod loss;
