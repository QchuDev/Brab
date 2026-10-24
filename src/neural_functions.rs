use std::collections::HashMap;

use nalgebra::DVector;

// ==================================== REGISTRY =========================================||
pub struct FunctionsRegistry {
    activation_functions: HashMap<String, Box<dyn ActivationFunction>>,
    cost_functions: HashMap<String, Box<dyn CostFunction>>,
}
impl FunctionsRegistry {
    pub fn new() -> Self {
        Self {
            activation_functions: HashMap::new(),
            cost_functions: HashMap::new(),
        }
    }

    /// Default registry of built-in functions
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.add_activation(Box::new(Sigmoid::new()));
        registry.add_activation(Box::new(SoftMax::new()));
        registry.add_cost(Box::new(CrossEntropy::new()));
        registry.add_cost(Box::new(Mse::new()));
        registry
    }

    pub fn add_activation(&mut self, new_activation: Box<dyn ActivationFunction>) {
        self.activation_functions
            .insert(new_activation.name(), new_activation);
    }

    pub fn add_cost(&mut self, new_cost: Box<dyn CostFunction>) {
        self.cost_functions.insert(new_cost.name(), new_cost);
    }

    pub fn get_activation(&self, alias: &str) -> Option<&dyn ActivationFunction> {
        self.activation_functions.get(alias).map(|b| b.as_ref())
    }

    pub fn get_cost(&self, alias: &str) -> Option<&dyn CostFunction> {
        self.cost_functions.get(alias).map(|b| b.as_ref())
    }
}

impl Default for FunctionsRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}
// ==================================== DOMAIN =========================================||
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
    fn name(&self) -> String;
}

pub trait CostFunction {
    /// Search it up, its used for the backpropagation algorithm
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64>;
    fn name(&self) -> String;
}

// ==================================== DEFAULT ACTIVATION FUNCTIONS ==========================================||
pub struct Sigmoid {}
impl Sigmoid {
    pub const fn new() -> Self {
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
    fn name(&self) -> String {
        "sigmoid".to_string()
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
    fn name(&self) -> String {
        "softmax".to_string()
    }
}

// ==================================== DEFAULT COST FUNCTIONS ==========================================||
pub struct Mse {}
impl Mse {
    pub const fn new() -> Self {
        Self {}
    }
}
impl CostFunction for Mse {
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        let sigmap_z = context.activ_fn.prime(context.z.clone());
        (context.a - context.y).component_mul(&sigmap_z)
    }

    fn name(&self) -> String {
        "mean square error".to_string()
    }
}

pub struct CrossEntropy {}
impl CrossEntropy {
    pub const fn new() -> Self {
        Self {}
    }
}
impl CostFunction for CrossEntropy {
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        context.a - context.y
    }

    fn name(&self) -> String {
        "cross entropy".to_string()
    }
}
