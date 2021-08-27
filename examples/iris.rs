use anyhow::Result;
use arrayfire as af;
use ox::{linear_model::LogisticRegression, BaseModel};
use oxide as ox;
use polars::prelude::*;

fn main() -> Result<()> {
    // pick which backend to use
    ox::set_backend("CPU");

    // set the random seed
    ox::set_seed(42u64);

    // load the iris dataset as a polars DataFrame
    let iris_dataset: DataFrame = ox::datasets::iris()?;

    // make an array fire dataset from the polars DataFrame
    let (x, targets): (af::Array<f32>, af::Array<f32>) =
        ox::preprocessing::make_dataset(iris_dataset, "Target")?;

    // create a LogisticClassifier and fit it to the dataset
    let mut model: LogisticRegression = LogisticRegression::new("sgd");
    model.fit(10, x, targets, 1e-3, 10);
    Ok(())
}
