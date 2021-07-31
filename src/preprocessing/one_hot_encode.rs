use anyhow::Result;
use polars::prelude::DataFrame;

/// One hot encoding for categorical columns
///
/// # Parameters:
/// - datafrmae: dataframe to use encode
/// - columns: the columns in the dataframe to encode
///
/// # Return Values
/// - dataframe with columns one hot encoded
pub fn one_hot_encode(mut dataframe: DataFrame, columns: Vec<&str>) -> Result<DataFrame> {
    let dummies = dataframe.select(&columns)?.to_dummies()?;
    let dummy_col_names = dummies.get_column_names();
    let test_series = dummies.select_series(dummy_col_names)?;
    for col in columns {
        dataframe = dataframe.drop(col)?;
    }
    for col in test_series {
        dataframe.with_column(col)?;
    }
    Ok(dataframe)
}
