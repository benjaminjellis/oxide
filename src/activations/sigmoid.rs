/// Sigmoid
///
/// Element-wise sigmoid
/// # Parameters
///
/// - input: an array
///
/// # Return Values
/// An array
pub fn sigmoid(input: &af::Array<f32>) -> af::Array<f32> {
    af::sigmoid(input)
}
