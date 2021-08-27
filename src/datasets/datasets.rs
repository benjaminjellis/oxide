use anyhow::Result;
use polars::prelude::*;

/// IRIS Dataset
pub fn iris() -> Result<DataFrame> {
    let df: DataFrame = CsvReader::from_path("./src/datasets/data/iris.csv")?
        .infer_schema(None)
        .has_header(true)
        .finish()?;
    Ok(df)
}

/// Titanic Dataset
///
/// # Parameters:
/// - split: which split to return, "test" or "train"
pub fn titanic(split: &str) -> Result<DataFrame> {
    if split == "train" {
        let df: DataFrame = CsvReader::from_path("./src/datasets/data/titanic_train.csv")?
            .infer_schema(None)
            .has_header(true)
            .finish()?;
        Ok(df)
    } else if split == "test" {
        let df: DataFrame = CsvReader::from_path("./src/datasets/data/titanic_test.csv")?
            .infer_schema(None)
            .has_header(true)
            .finish()?;
        Ok(df)
    } else {
        panic!("Expected split to be one of test or train, got {}", split);
    }
}

/// Binary Classification Test Dataset
/// # Parameters:
/// - split: which split to return, "test" or "train"
pub fn bc_test_dataset(split: &str) -> Result<DataFrame> {
    if split == "train" {
        let df: DataFrame = CsvReader::from_path("./src/datasets/data/bc_test_data_train.csv")?
            .infer_schema(None)
            .has_header(true)
            .finish()?;
        Ok(df)
    } else if split == "test" {
        let df: DataFrame = CsvReader::from_path("./src/datasets/data/bc_test_data_test.csv")?
            .infer_schema(None)
            .has_header(true)
            .finish()?;
        Ok(df)
    } else {
        panic!("Expected split to be one of test or train, got {}", split);
    }
}
