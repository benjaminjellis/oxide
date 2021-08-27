![build status](https://github.com/benjaminjellis/oxide/actions/workflows/rust.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/benjaminjellis/oxide/blob/main/LICENSE)

[<img align="right" src="docs/static/logo.png" width="300">](#)
# oxide 
experimental library for simple, high-performance ML in Rust with hardware acceleration
## Setup

oxide uses [ArrayFire](https://github.com/arrayfire/arrayfire-rust) as a backend, to use ArrayFire you first need to install the binaries.
Instructions for this can be found [here](https://github.com/arrayfire/arrayfire-rust#use-from-cratesio--). Using ArrayFire means oxide can run on OpenCL, CUDA and CPU devices.

## Getting Started

### Loading Data 

oxide integrates with [polars](https://docs.rs/polars/0.14.2/polars/) so loading and preprocessing data is easy

Load a toy dataset:
```rust
use oxide as ox;

// get the titanic train and test datasets as polars dataframe
let train_df = ox::datasets::titanic(&"train");
let test_df = ox::datasets::titanic(&"test");

```

Load your own dataset from csv:

```rust
use polars::prelude::*;
let df = CsvReader::from_path("your_data.csv")?
            .infer_schema(None)
            .has_header(true)
            .finish()?;
```

Convert your dataset to an oxide dataset:

```rust
let (x, targets) = ox::preprocessing::make_dataset(train_dataset, "Target")?;
```

### Training a model

```rust 
let mut model = ox::linear_model::LogisticRegression::new("sgd");
model.fit(10u16, x, targets, 0.001, 10)?;
``` 

For more examples see [here](https://github.com/benjaminjellis/oxide/tree/main/examples).
## Contributions

oxide is under active development, and a list of features to build for v0.1.0 is available in the issues of this repository. 
Contributions are very welcome, please see the [CONTRIBUTING.md](https://github.com/benjaminjellis/oxide/blob/main/CONTRIBUTING.md).
