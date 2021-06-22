use crate::function::*;
use std::f64::consts::PI;

struct Rastrigin {
    a: f64,
    n: usize,
}


impl Rastrigin {
    pub fn new(n: usize, a: f64) -> Self {
        Self { a, n }
    }
}

impl BlackBoxFunction for Rastrigin {
    fn f(&self, x: Vec<f64>) -> Vec<f64> {
        let mut y = 0.;
        for x_i in x {
            y += (x_i.powi(2) - self.a * (2 * PI * x_i).cos())
        }
        y + self.a * self.n
    }

    fn get_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut upper: Vec<f64> = vec![];
        let mut lower: Vec<f64> = vec![];
        for _ in 0..self.n {
            upper.push(5.12);
            lower.push(-5.12);
        }
        (lower, upper)
    }

    fn get_x_dim(&self) -> usize {
        self.n
    }
}