use anyhow::Result;

/// Base for all models
pub trait BaseModel {
    // TODO add more signatures, serlisation for saving and loading model, plus
    /// Basic datasets validation
    fn validate_data(x: &af::Array<f32>, y: &af::Array<f32>) {
        assert_eq!(
            x.dims()[0],
            y.dims()[0],
            "Data validation failed, check that the nuber of \
    observations, is equal to the number of labels"
        );
    }

    /// Signature for fit function
    /// # Parameters
    ///
    /// - epochs: number of epochs to run for
    /// - x: features of shape (m, n)
    /// - y: labels of shape (m, 1)
    ///
    /// # Return Values
    /// nothing
    fn fit(
        &mut self,
        epochs: u16,
        x: af::Array<f32>,
        targets: af::Array<f32>,
        lr: f32,
        batch_size: u64,
    ) -> Result<()>;

    /// Signature for predict function
    ///
    /// # Parameters
    /// - x: features of shape (m, n)
    /// # Return Values
    /// predictions of shape (m, 1)
    fn predict(&mut self, x: af::Array<f32>) -> Result<af::Array<f32>>;

    // fn save_state_dict(path: &'static str)

    // fn load_state_dict(path: &'static str)

    // fn evaluate
}
