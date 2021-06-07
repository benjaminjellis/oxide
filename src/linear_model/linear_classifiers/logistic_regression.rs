use crate::{activations, base, loss};
use af;

/// Logistic Regression Model
///
/// # Examples
///
/// ```rust
/// use oxide::{base, linear_model};
/// // create a logistic regression model
/// let mut model = linear_model::LogisticRegression::new(&"sgd");
/// // fit the model for 10 epochs on your data x and y
/// model.fit(10, &x, &y);
/// ```
pub struct LogisticRegression {
    classes: af::Array<f32>,
    w: af::Array<f32>,
    b: af::Array<f32>,
    epochs: u16,
    solver: &'static str,
    batch_size: u16
}

impl LogisticRegression {
    /// Create a new logistic regression model
    ///
    /// # Parameters
    ///
    /// - solver: which solver to use, currently only "sgd" is implemented
    ///
    /// # Return Values
    /// nothing
    ///
    pub fn new(solver: &'static str) -> LogisticRegression {
        base::which_backend();
        LogisticRegression {
            classes: af::randn!(1),
            w: af::randn!(1),
            b: af::randn!(1),
            epochs: 100,
            solver: solver,
            batch_size: 120
        }
    }

    fn binary_sgd(&mut self) {

    }
}

impl base::BaseModel for LogisticRegression {
    /// Fit the model
    ///
    /// # Parameters
    ///
    /// - epochs: number of epochs to run for
    /// - x: features of shape (m, n)
    /// - y: labels of shape (m, 1)
    ///
    /// # Return Values
    /// nothing
    fn fit(&mut self, epochs: u16, x: af::Array<f32>, y: af::Array<f32>) {
        self.epochs = epochs;
        //self.batch_size = batch_size;

        LogisticRegression::validate_data(&x, &y);

        self.w = af::randn!(x.dims()[1]);
        self.b = af::randn!(x.dims()[0]);
        self.classes = af::set_unique(&y, false);
        let mut n_classes = self.classes.dims()[0];

        let mut classes = &self.classes;

        if n_classes == 2 {
            // binary classification

            let preds = activations::sigmoid(&af::add(
                &af::matmul(&x, &self.w, af::MatProp::NONE, af::MatProp::NONE),
                &self.b,
                false,
            ));

            if self.solver == "sgd" {

                LogisticRegression::binary_sgd(self);

                // do SGD here
            } else {
                panic!("Currently only sgd is a supported solver")
            }
        } else if n_classes > 2 {
            // mutliclass
            panic!("This model currently only supports two class classification, ")
        } else {
            panic!(
                "Supplied data must contain observations of two or more classes, \
            found {} classes",
                n_classes
            )
        }
    }

    /// Make predictions
    ///
    /// # Parameters
    ///
    /// - x: features of shape (m, n)
    ///
    /// # Return Values
    /// predictions of shape (m, 1)
    fn predict(&mut self, x: af::Array<f32>) -> af::Array<f32> {
        let mut preds = activations::sigmoid(&af::add(
                &af::matmul(&x, &self.w, af::MatProp::NONE, af::MatProp::NONE),
                &self.b,
                false,
            ));

        let preds = af::round(&preds);

        return preds;
    }
}
