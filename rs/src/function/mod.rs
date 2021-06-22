mod benchmark;

pub trait BlackBoxFunction {
    /// Evaluates for function response.
    fn f(&self, x: Vec<f64>) -> Vec<f64>;

    /// Retrieves bounds of input domain.
    fn get_bounds(&self) -> Vec<Vec<f64>>;

    /// Retrieve function input dimension.
    fn get_x_dim(&self) -> usize;
}