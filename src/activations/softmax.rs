#[allow(dead_code)]
/// Softmax
///
/// softmax applied over a dimension
/// # Parameters
///
/// - input: an array
/// - dim: dimension to sum over
///
/// # Return Values
/// An array
pub fn softmax(input: &af::Array<f32>, dim: i32) -> af::Array<f32> {
    let exponential_el = af::exp(input);
    let exponential_el_sum = af::sum(&exponential_el, dim);
    af::div(&exponential_el, &exponential_el_sum, true)
}
