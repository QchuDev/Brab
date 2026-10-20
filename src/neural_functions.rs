use nalgebra::DVector;

pub struct NeuralFunctionContext<'a> {
    pub z: &'a DVector<f64>,
    pub a: &'a DVector<f64>,
    pub y: &'a DVector<f64>,
    pub cost_fn: &'a dyn CostFunction,
    pub activ_fn: &'a dyn ActivationFunction,
}

pub trait ActivationFunction {
    fn fun(&self, x: DVector<f64>) -> DVector<f64>;
    fn prime(&self, x: DVector<f64>) -> DVector<f64>;
}

pub trait CostFunction {
    /// Search it up, its used for the backpropagation algorithm
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64>;
}

// ==================================== IMPLEMENTATIONS ==========================================||

// ==================================== ACTIVATION FUNCTIONS ==========================================||
pub struct Sigmoid {}
impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }
}
impl ActivationFunction for Sigmoid {
    fn fun(&self, x: DVector<f64>) -> DVector<f64> {
        x.map(|x_i| sigmoid(x_i))
    }
    fn prime(&self, x: DVector<f64>) -> DVector<f64> {
        x.map(|x_i| sigmoid_prime(x_i))
    }
}
fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (-x).exp())
}
fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1f64 - sigmoid(x))
}

pub struct SoftMax {}
impl SoftMax {
    pub const fn new() -> Self {
        Self {}
    }
}
impl ActivationFunction for SoftMax {
    fn fun(&self, x: DVector<f64>) -> DVector<f64> {
        x.map(|x_i| x_i.exp() / (x.map(|x_i| x_i.exp()).sum()))
    }
    fn prime(&self, x: DVector<f64>) -> DVector<f64> {
        self.fun(x.clone())
            .component_mul(&(DVector::repeat(x.nrows(), 1.0) - &self.fun(x)))
    }
}

// ==================================== COST FUNCTIONS ==========================================||
pub struct Mse {}
impl Mse {
    pub fn new() -> Self {
        Self {}
    }
}
impl CostFunction for Mse {
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        let sigmap_z = context.activ_fn.prime(context.z.clone());
        (context.a - context.y).component_mul(&sigmap_z)
    }
}

pub struct CrossEntropy {}
impl CrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}
impl CostFunction for CrossEntropy {
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        context.a - context.y
    }
}
