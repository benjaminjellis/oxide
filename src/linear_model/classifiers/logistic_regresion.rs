use crate::{activations, base, utils};
use anyhow::Result;

/// Logistic Regression Model
///
/// # Examples
///
/// ```rust
/// use oxide as ox;
/// use arrayfire as af;
/// // create random x and y
/// let x: af::Array<f32> = af::randn!(1000, 10);
/// let y: af::Array<f32> = af::randn!(100);
/// let unseen_x: af::Array<f32> = af::randn!(400, 10);
/// // create a logistic regression model
/// let mut model = ox::linear_model::LogisticRegression::new("sgd");
/// // fit the model for 10 epochs on your datasets x and y
/// model.fit(10, &x, &y, 1e-3);
/// // make inference on unseen datasets
/// predictions: af::Array<f32> = model.predict(&unssen_x);
/// ```
pub struct LogisticRegression {
    classes: af::Array<f32>,
    w: af::Array<f32>,
    b: f32,
    epochs: u16,
    lr: f32,
    solver: &'static str,
    batch_size: u64,
}

impl LogisticRegression {
    /// Create a new binary or multinomial logistic regression model
    ///
    /// # Parameters
    /// - solver: which solver to use, currently only "sgd" is supported
    ///
    /// # Return Values
    /// nothing
    ///
    pub fn new(solver: &'static str) -> LogisticRegression {
        //base::which_backend();
        LogisticRegression {
            classes: af::randn!(1),
            w: af::randn!(1),
            b: 0.0,
            epochs: 100,
            lr: 1e-3,
            solver,
            batch_size: 120,
        }
    }

    /// Binary SGD solver for Logistic Regression
    ///
    /// # Parameters
    ///
    /// # Return Values
    #[allow(clippy::ptr_arg)]
    fn _binary_sgd(&mut self, x_b: &Vec<af::Array<f32>>, y_b: &Vec<af::Array<f32>>) {
        for _epoch in 0..self.epochs {
            for mb in x_b.iter().zip(y_b.iter()) {
                let (x_mb, y_mb) = mb;
                let m: u64 = x_mb.dims()[0];
                let preds: af::Array<f32> = activations::sigmoid(
                    &(&af::matmul(x_mb, &self.w, af::MatProp::NONE, af::MatProp::NONE) + self.b),
                );

                // TODO dw is just array of zeros, why? the issue is multiplying the result of matmul and
                // (1u64 / m)
                let dw: af::Array<f32> = (1f32 / m as f32)
                    * af::matmul(
                        x_mb,
                        &af::sub(&preds, y_mb, false),
                        af::MatProp::TRANS,
                        af::MatProp::NONE,
                    );

                let db: af::Array<f32> = (1u64 / m) * af::sub(&preds, y_mb, false);

                self.w = af::sub(&self.w, &(self.lr * dw), false);
                self.b -= self.lr * utils::get_val(&db);
            }
            // TODO what about loss after &epoch epochs here? this could be done by calcing accuracy
        }
    }

    /// Multinomial SGD solver for Logistic Regression
    #[allow(clippy::ptr_arg)]
    fn _multinomial_sgd(&mut self, x_b: &Vec<af::Array<f32>>, y_b: &Vec<af::Array<f32>>) {
        for _epoch in 0..self.epochs {
            for mb in x_b.iter().zip(y_b.iter()) {
                let (_x_mb, _y_mb) = mb;
            }
        }
    }
}

impl base::BaseModel for LogisticRegression {
    /// Fit the model, note: shuffling and batching is handled by this method
    ///
    /// # Parameters
    ///
    /// - epochs: number of epochs to run for
    /// - epochs: number of epochs to train for
    /// - x: features of shape (m, n)
    /// - targets: targets of shape (m, 1)
    /// - lr : learning rate
    /// - batch_size: batch size
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
    ) -> Result<()> {
        self.epochs = epochs;
        self.lr = lr;
        self.batch_size = batch_size;

        LogisticRegression::validate_data(&x, &targets);

        self.w = af::randn!(x.dims()[1]);
        self.classes = af::set_unique(&targets, false);
        let n_classes = self.classes.dims()[0];

        let _classes = &self.classes;

        if self.batch_size > x.dims()[0] {
            panic!("Batch size greater than the number of observations")
        } else {
            let (x_b, t_b) = utils::batch((&x, &targets), self.batch_size as u64);
            match n_classes {
                2 => match self.solver {
                    "sgd" => LogisticRegression::_binary_sgd(self, &x_b, &t_b),
                    _ => panic!("Currently only sgd is a supported solver for binary logistic regression")
                }
                nc if nc > 2 => match self.solver {
                    "sgd" => LogisticRegression::_multinomial_sgd(self, &x_b, &t_b),
                    _ => panic!("Currently only sgd is a supported solver for multinomial logistic regression")
                }
                _ => panic!(
                    "Supplied datasets must contain observations of two or more classes, \
                    found {} classes",
                    n_classes)
            }
        }
        Ok(())
    }

    /// Make predictions
    ///
    /// # Parameters
    ///
    /// - x: features of shape (m, n)
    ///
    /// # Return Values
    /// predictions of shape (m, 1)
    fn predict(&mut self, x: af::Array<f32>) -> Result<af::Array<f32>> {
        let preds = activations::sigmoid(&af::add(
            &af::matmul(&x, &self.w, af::MatProp::NONE, af::MatProp::NONE),
            &self.b,
            false,
        ));
        Ok(af::round(&preds))
    }
}
