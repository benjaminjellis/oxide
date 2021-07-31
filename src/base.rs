use std::process;

/// Base for all models
pub trait BaseModel {
    // TODO add more signatures, maybe fit and predict should be signatures?
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
        epcohs: u16,
        x: af::Array<f32>,
        targets: af::Array<f32>,
        lr: f32,
        batch_size: u64,
    );

    /// Signature for fit function
    ///
    /// # Parameters
    /// - x: features of shape (m, n)
    /// # Return Values
    /// predictions of shape (m, 1)
    fn predict(&mut self, x: af::Array<f32>) -> af::Array<f32>;

    // fn save_state_dict(path: &'static str)

    // fn load_state_dict(path: &'static str)
}

/// Find what backend are available and set in order of preference
pub fn which_backend() {
    // TODO rexport this somwhere else, make it so you have to set it first thing
    let available_backends = af::get_available_backends();

    if available_backends.contains(&af::Backend::CUDA) {
        println!("CUDA backend found, using CUDA");
        af::set_backend(af::Backend::CUDA);
        println!("There are {} CUDA device(s)", af::device_count());
    } else if available_backends.contains(&af::Backend::CPU) {
        println!("CPU backend found, using CPU");
        af::set_backend(af::Backend::CPU);
        println!("There are {} CPU device(s)", af::device_count());
    } else {
        println!("No compatible backend found, was looking for CPU or CUDA");
        println!("Exiting process");
        process::exit(1);
    };
}
