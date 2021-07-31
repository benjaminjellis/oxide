use crate::{activations, utils};

/// Binary Cross Entropy Loss
///
/// # Parameters
///
/// - y: ground truth labels of shape (m, 1)
/// - target: predicted labels of shape (m, 1)
///
/// # Return Values
/// loss of shape (1)
pub fn binary_cross_entropy(y: &af::Array<f32>, target: &af::Array<f32>) -> af::Array<f32> {
    let pred = activations::sigmoid(y);
    let eps = 1e-10;
    let pos = af::mul(
        &af::mul(&-1.0f32, target, false),
        &af::log(&utils::clip_by_value(&pred, eps, 1.0 - eps)),
        false,
    );
    let neg = af::mul(
        &af::sub(&1.0f32, target, false),
        &af::log(&utils::clip_by_value(
            &af::sub(&1.0f32, &pred, false),
            eps,
            1.0 - eps,
        )),
        false,
    );

    af::mean(&af::sub(&pos, &neg, false), 0)
}
