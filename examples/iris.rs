use arrayfire as af;
use ox::{linear_model, BaseModel};
use oxide as ox;

fn main() {
    let (x, y) = ox::Datasets::iris();
    let mut model = linear_model::LogisticRegression::new(&"sgd");
    model.fit(10, &x, &y, 1e-3);
}
