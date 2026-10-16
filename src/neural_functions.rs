use nalgebra::DVector;

pub struct NeuralFunctionContext<'a> {
    pub z: &'a DVector<f64>,
    pub a: &'a DVector<f64>,
    pub y: &'a DVector<f64>,
    pub cost_fn: &'a dyn CostFunction,
    pub activ_fn: &'a dyn ActivationFunction,
}

pub trait ActivationFunction {
    fn fun(&self, x: f64) -> f64;
    fn prime(&self, x: f64) -> f64;
}

pub trait CostFunction {
    fn fun(&self, context: NeuralFunctionContext) -> f64;

    /// Search it up, its used for the backpropagation algorithm
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64>;
}

pub struct Relu {}
impl Relu {
    pub fn new() -> Self {
        Self {}
    }
}
impl ActivationFunction for Relu {
    fn fun(&self, x: f64) -> f64 {
        if x < -1.0 { 0.0 } else { x }
    }
    fn prime(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0;
        }
        0.0
    }
}

pub struct Sigmoid {}
impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }
}
impl ActivationFunction for Sigmoid {
    fn fun(&self, x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }
    fn prime(&self, x: f64) -> f64 {
        self.fun(x) * (1f64 - self.fun(x))
    }
}

/// This CostFunction trait expects you to implement: the function itself and its gradient respect to a (activation)

pub struct Mse {}
impl Mse {
    pub fn new() -> Self {
        Self {}
    }
}
impl CostFunction for Mse {
    fn fun(&self, context: NeuralFunctionContext) -> f64 {
        if context.y.nrows() != context.a.nrows() {
            return 0f64;
        }

        let n: usize = context.y.nrows();
        let mut sum: f64 = 0.0;

        for i in 0..n {
            sum += (context.y[i] - context.a[i]).powi(2);
        }

        return sum / (n as f64);
    }

    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        let sigmap_z = DVector::from_fn(context.z.nrows(), |i, _j| {
            (context.activ_fn).prime(context.z[i])
        });

        (context.a - context.y).component_mul(&sigmap_z)
    }
}

// WIP
pub struct CrossEntropy {}
impl CrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}
impl CostFunction for CrossEntropy {
    fn fun(&self, context: NeuralFunctionContext) -> f64 {
        let mut c = 0.0;
        let n = context.a.nrows();

        for (j, a_j) in context.a.iter().enumerate() {
            c += context.y[j] * a_j.ln() + (1.0 - context.y[j]) * (1.0 - a_j).ln();
        }

        c *= -1.0 / n as f64;
        c
    }

    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        context.a - context.y
    }
}
