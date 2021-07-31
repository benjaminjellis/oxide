use anyhow::Result;
use oxide as ox;

pub fn main() -> Result<()> {
    // load the titanic train dataset
    let mut df = ox::datasets::titanic(&"train")?;
    // use one-hot encoding
    df = ox::preprocessing::one_hot_encode(df, vec!["Embarked", "Sex", "Pclass"])?;
    Ok(())
}
