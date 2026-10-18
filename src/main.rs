use brab::{
    network::{NetworkConfig, NeuralNetwork},
    neural_functions::{CrossEntropy, Sigmoid},
};
use nalgebra::DVector;

include!("mnist_data.rs");

fn main() {
    println!("Start..\n");

    // Get data from MNIST data-set
    let mnist_prov = MnistProvider::new();
    let train_data = &mnist_prov.get_data();

    // Define the functions to use
    let sigmoid = Sigmoid::new(); // Activation function

    let cross_entropy = CrossEntropy::new();

    let net_shape: Vec<usize> = vec![784, 8, 5, 10];

    let config_netce = NetworkConfig {
        layers_sizes: &net_shape,
        new_act_fn: &sigmoid,
        new_cst_fn: &cross_entropy,
        has_softmax: true,
    };

    // Create network
    let mut net_ce: NeuralNetwork = NeuralNetwork::new_load(
        "models/net_ce",
        config_netce.new_act_fn,
        config_netce.new_cst_fn,
        true,
    )
    .unwrap();

    // Run for 3 epochs
    for _ in 0..1 {
        println!(
            "{}",
            net_ce.train_multiple(0.06f64, train_data.clone()).unwrap()
        );
    }

    let _ = net_ce.save("models/net_ce");
}
