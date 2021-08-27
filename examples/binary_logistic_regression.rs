use anyhow::Result;
use arrayfire as af;
use oxide as ox;
use ox::{linear_model::LogisticRegression, BaseModel};
use polars::prelude::*;
use std::time::Instant;


fn main() -> Result<()> {
    let now = Instant::now();

    // set backend
    ox::set_backend("CPU");

    // set random seed
    ox::set_seed(2u64);

    // create a train dataset
    let train_dataset: DataFrame = ox::datasets::bc_test_dataset("train")?;
    let (x, targets): (af::Array<f32>, af::Array<f32>) =
        ox::preprocessing::make_dataset(train_dataset, "Target", false, 0f32)?;

    // create a test dataset
    let test_dataset: DataFrame = ox::datasets::bc_test_dataset("test")?;
    let (x_test, targets_test): (af::Array<f32>, af::Array<f32>) =
        ox::preprocessing::make_dataset(test_dataset, "Target", false, 0f32)?;

    // create a LogisticClassifier model and fit it
    let mut model: LogisticRegression = LogisticRegression::new("sgd");
    model.fit(25u16, x, targets, 0.01, 50)?;

    // get the predictions on the test dataset and calculate the accuracy
    let predictions: af::Array<f32> = model.predict(x_test)?;
    let accuracy: af::Array<f32> = ox::metrics::accuracy(&predictions, &targets_test);
    af::af_print!("Test Accuracy: ", accuracy);

    println!("Time taken: {} (s)", now.elapsed().as_secs());
    Ok(())
}
