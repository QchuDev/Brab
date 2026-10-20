use crate::{
    consts::{CROSS_ENTROPY, SIGMOID},
    neural_functions::{ActivationFunction, CostFunction},
};

/*
    TODO: Refactor the way of creating a NeuralNetwork:
    - NeuralNetowork will start default
    - No config is provided
    - by calling chained methods (reactive) it will set itself
    - Method for each config field
*/

pub struct NetworkConfig<'a> {
    pub layers_sizes: &'a [usize],
    pub new_act_fn: &'a dyn ActivationFunction,
    pub new_cst_fn: &'a dyn CostFunction,
    pub has_softmax: bool,
}
impl<'a> NetworkConfig<'a> {
    pub fn new() -> Self {
        NetworkConfig {
            layers_sizes: &[4, 3, 2],
            new_act_fn: &SIGMOID,
            new_cst_fn: &CROSS_ENTROPY,
            has_softmax: true,
        }
    }

    pub fn with_shape(mut self, shape: &'a [usize]) -> Self {
        self.layers_sizes = shape;
        self
    }

    pub fn with_activation(mut self, activation: &'a dyn ActivationFunction) -> Self {
        self.new_act_fn = activation;
        self
    }

    pub fn with_cost(mut self, cost: &'a dyn CostFunction) -> Self {
        self.new_cst_fn = cost;
        self
    }

    pub fn with_sofmax(mut self) -> Self {
        self.has_softmax = true;
        self
    }

    pub fn build(self) -> Self {
        self
    }
}
