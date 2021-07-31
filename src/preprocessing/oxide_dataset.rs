use anyhow::Result;
use polars::datatypes::Float32Type;
use polars::prelude::DataFrame;

/// Convert a polars dataframe into an oxide dataset
///
/// # Parameters
/// - dataframe: polars dataframe to extract data from
/// - target_column: name of the column that contains the target
///
/// # Returns
/// - (features, targets) a tuple of af::Array<f32> arrays
#[allow(unused_must_use)]
pub fn make_dataset(
    mut dataframe: DataFrame,
    target_column: &str,
) -> Result<(af::Array<f32>, af::Array<f32>)> {
    // create target array
    let target_series = &dataframe.select_series(target_column)?[0].cast::<Float32Type>()?;

    if target_series.null_count() != 0 {
        panic!("Cannot create an oxide dataset with null values, encountered nulls when creating the targets array")
    }

    dataframe.drop_in_place(target_column);

    let target_shape = target_series.len() as u64;
    let target_dim4 = af::Dim4::new(&[target_shape, 1, 1, 1]);
    let mut target_values = vec![];
    let target_values_ca = target_series.unpack::<Float32Type>()?;

    target_values_ca
        .into_no_null_iter()
        .enumerate()
        .for_each(|(_row_idx, val)| {
            target_values.push(val);
        });

    let targets = af::Array::new(&target_values, target_dim4);

    let mut feature_values = vec![];
    let (n_obs, no_features) = dataframe.shape();
    let features_dim4 = af::Dim4::new(&[n_obs as u64, no_features as u64, 1, 1]);
    // create features
    for (_col_idx, series) in dataframe.get_columns().iter().enumerate() {
        if series.null_count() != 0 {
            panic!("Cannot create an oxide dataset with null values, encountered nulls when creating the features array")
        }
        // this is an Arc clone if already of type N
        let series = series.cast::<Float32Type>()?;
        let ca = series.unpack::<Float32Type>()?;

        ca.into_no_null_iter()
            .enumerate()
            .for_each(|(_row_idx, val)| {
                feature_values.push(val);
            });
    }
    let features = af::Array::new(&feature_values, features_dim4);
    Ok((features, targets))
}
