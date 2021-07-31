/// Mean Squared Error Loss
///
/// # Parameters
/// - y: ground truth labels of shape (m, 1)
/// - target: predicted labels of shape (m, 1)
///
/// # Return Values
/// loss of shape (1)
pub fn mse(y: af::Array<f32>, target: af::Array<f32>) -> af::Array<f32> {
    let term: af::Array<f32> = af::pow(&af::sub(&y, &target, false), &2, false);
    af::mean(&term, 0)
}
