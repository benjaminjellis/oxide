/// Calculate accuracy for two 1-d arrays
///
/// # Parameters
/// - y: predicted labels of shape (m, 1)
/// - target: ground truth labels of shape (m, 1)
///
/// # Return Values
/// accuracy, expressed as decimal percentage
pub fn accuracy(y: &af::Array<f32>, target: &af::Array<f32>) -> af::Array<f32> {
    assert_eq!(
        y.dims(),
        target.dims(),
        "Expected y and target to have the same dims but got {} and {}",
        y.dims(),
        target.dims()
    );
    let eq = af::eq(y, target, false);
    let correct = af::count(&eq, 0).cast::<f32>();
    correct / y.dims()[0]
}
