/// Relu
///
/// Element-wise rectified-linear unit
/// # Parameters
///
/// - input: an array
///
/// # Return Values
/// An array
pub fn relu(input: &af::Array<f32>) -> af::Array<f32> {
    let zero = af::constant(0.01f32, input.dims());
    af::select(&zero, &af::lt(input, &0.0, false), input)
}
