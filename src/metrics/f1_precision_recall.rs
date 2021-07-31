use crate::metrics::confusion_matrix;

/// F1 Score
/// - y: predicted labels of shape (m, 1)
/// - target: ground truth labels of shape (m, 1)
///
pub fn f1(y: &af::Array<f32>, target: &af::Array<f32>) -> af::Array<f32> {
    let cm = confusion_matrix(y, target);
    let precision: af::Array<f32> = _precision(&cm);
    let recall: af::Array<f32> = _recall(&cm);
    2 * (&precision * &recall) / &precision - &recall
}

/// Calculate precision for two 1-d arrays (currently only binary)
///
/// - y: predicted labels of shape (m, 1)
/// - target: ground truth labels of shape (m, 1)
///
pub fn precision(y: &af::Array<f32>, target: &af::Array<f32>) -> af::Array<f32> {
    let cm = confusion_matrix(y, target);
    _precision(&cm)
}

/// private helper function
fn _precision(cm: &af::Array<f32>) -> af::Array<f32> {
    let tp_idx = &[af::Seq::new(0u32, 0, 1), af::Seq::new(0u32, 0, 1)];
    let fp_idx = &[af::Seq::new(1u32, 1, 1), af::Seq::new(0u32, 0, 1)];
    let tp = af::index(cm, tp_idx);
    let fp = af::index(cm, fp_idx);
    &tp / (&tp + &fp)
}

/// Calculate recall for two 1-d arrays (currently only binary)
///
/// - y: predicted labels of shape (m, 1)
/// - target: ground truth labels of shape (m, 1)
///
pub fn recall(y: &af::Array<f32>, target: &af::Array<f32>) {
    let cm = confusion_matrix(y, target);
    _recall(&cm);
}

/// private helper function
fn _recall(cm: &af::Array<f32>) -> af::Array<f32> {
    let tp_idx = &[af::Seq::new(0u32, 0, 1), af::Seq::new(0u32, 0, 1)];
    let fn_idx = &[af::Seq::new(0u32, 0, 1), af::Seq::new(1, 1, 0)];
    let tp = af::index(cm, tp_idx);
    let fan = af::index(cm, fn_idx);
    &tp / (&tp + &fan)
}
