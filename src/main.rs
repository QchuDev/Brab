use brab::{
    network::{NetworkConfig, NeuralNetwork},
    neural_functions::{CrossEntropy, Sigmoid},
    training::{SGD, Trainer},
};
use nalgebra::DVector;

include!("mnist_data.rs");

fn main() {
    println!("Start..\n");

    // Get data from MNIST data-set
    let mnist_prov = MnistProvider::new();
    let train_data = mnist_prov.get_data();

    // Define the functions to use
    let sigmoid = Sigmoid::new(); // Activation function

    let cross_entropy = CrossEntropy::new();

    let net_shape: Vec<usize> = vec![784, 8, 5, 10];

    let config = NetworkConfig {
        layers_sizes: &net_shape,
        new_act_fn: &sigmoid,
        new_cst_fn: &cross_entropy,
        has_softmax: true,
    };

    let mut net = NeuralNetwork::new(config);
    Trainer::train(&mut net, train_data, &SGD::new(0.06f64), 3);
}
