use crate::data::TrainData;
use crate::network::{NeuralNetwork, ParametersGradient};
use core::f64;

pub struct Trainer {}
impl Trainer {
    pub fn train(
        network: &mut NeuralNetwork,
        train_data: Vec<TrainData>,
        optimizer: &dyn Optimizer,
        epochs: i32,
    ) {
        for _e in 0..epochs {
            for td in &train_data {
                let _ = network.feed_forward(td.input.clone());

                let gradient = &network.backprop(td.expected.clone()).unwrap();
                optimizer.step(network, gradient);
                println!("{}", network.get_output());
            }
            println!("Processed {}/{} epochs", _e, epochs);
        }
    }
}

/// This trait is for those that update parameters inside the network
pub trait Optimizer {
    fn step(&self, network: &mut NeuralNetwork, gradient: &ParametersGradient);
}

// Stochastic Gradient Descent (Optimizer)
pub struct SGD {
    learning_rate: f64,
}
impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}
impl Optimizer for SGD {
    fn step(&self, network: &mut NeuralNetwork, gradient: &ParametersGradient) {
        for i in 0..network.weights.len() {
            let new_w = &network.weights[i] - self.learning_rate * &gradient.nabla_w[i];
            network.weights[i] = new_w;
        }

        for i in 0..network.layers.len() - 1 {
            network.layers[i + 1].b =
                &network.layers[i + 1].b - self.learning_rate * &gradient.nabla_b[i];
        }
    }
}
