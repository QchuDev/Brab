use crate::data::TrainData;
use crate::network::{NeuralNetwork, ParametersGradient};
use core::f64;
use std::fmt::{Display, Formatter};
pub struct Trainer {}
impl Trainer {
    pub fn train(
        network: &mut NeuralNetwork,
        train_data: Vec<TrainData>,
        optimizer: &dyn Optimizer,
        epochs: i32,
    ) -> Result<Vec<EpochResult>, String> {
        let mut results = Vec::new();
        for _e in 0..epochs {
            let mut epoch_res = vec![
                vec![0.0, 0.95],
                vec![0.0, 0.90],
                vec![0.0, 0.50],
                vec![0.0, 0.25],
            ];
            let n: f64 = train_data.len() as f64;

            for td in &train_data {
                // Feed
                let fed_result = network.feed_forward(td.input.clone());
                if let Err(s) = fed_result {
                    return Err(s.to_string());
                }

                // Calculate Parameters change
                let gradient = &network.backprop(td.expected.clone()).unwrap();

                optimizer.step(network, gradient);

                // FROM HERE ITS ALL RESULTS EXTRACTION
                let dot = td.expected.dot(network.get_output());

                epoch_res = epoch_res
                    .iter()
                    .map(|x| -> Vec<f64> {
                        let mut increment = 0.0;
                        if dot >= x[1] {
                            increment += 1.0;
                        }
                        vec![x[0] + increment, x[1]]
                    })
                    .collect();
            }
            epoch_res = epoch_res.iter().map(|e| vec![e[0] / n, e[1]]).collect();
            results.push(EpochResult {
                freq_over: epoch_res,
            });
            println!("Processed {}/{} epochs", _e + 1, epochs);
        }

        return Ok(results);
    }
}

pub struct EpochResult {
    freq_over: Vec<Vec<f64>>,
}
impl Display for EpochResult {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        let output: String = self
            .freq_over
            .iter()
            .map(|f_v| format!("(f={}, p={})\n", f_v[0], f_v[1]))
            .collect();

        write!(f, "{}", output)
    }
}

/// This traits for those that update parameters inside the network
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
        for i in 0..network.get_size() - 1 {
            let new_w = network.get_weight(i) - self.learning_rate * &gradient.nabla_w[i];
            network.set_weight(i, new_w);
        }

        for i in 0..network.get_size() - 1 {
            let new_b = network.get_bias(i + 1) - self.learning_rate * &gradient.nabla_b[i];
            network.set_bias(i + 1, new_b);
        }
    }
}
