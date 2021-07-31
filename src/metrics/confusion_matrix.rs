use crate::utils::get_val;

/// Create a confusion matrix for binary or multiclass classification
///
/// # Parameters
/// - y: predicted labels of shape (m, 1)
/// - target: ground truth labels of shape (m, 1)
///
/// # Return Values
/// confusion matrix of shape (n_classes, n_classes)
pub fn confusion_matrix(y: &af::Array<f32>, target: &af::Array<f32>) -> af::Array<f32> {
    let no_classes = af::set_unique(&target, false).dims()[0];
    let mut cm = af::constant!(0.0f32; no_classes, no_classes);

    for i in 0..(y.dims()[0]) as i64 {
        let y_i = get_val(&af::row(&y, i));
        let t_i = get_val(&af::row(&target, i));
        let indices = [af::Seq::new(t_i, t_i, 1f32), af::Seq::new(y_i, y_i, 1f32)];

        let current_val = get_val(&af::index(&cm, &indices));
        let new_val = current_val + 1f32;
        let array_replacement = af::constant(new_val as f32, af::dim4!(1));
        af::assign_seq(&mut cm, &indices, &array_replacement);
    }
    cm
}
