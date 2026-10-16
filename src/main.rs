use core::f64;

use brab::{
    network::NeuralNetwork,
    neural_functions::{CrossEntropy, Mse, Sigmoid},
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

    let mse = Mse::new(); // Cost functions
    let cross_entropy = CrossEntropy::new();

    // Load network
    let mut net_mse: NeuralNetwork =
        NeuralNetwork::new_load("models/model_1", &sigmoid, &mse).unwrap();

    let mut net_ce: NeuralNetwork =
        NeuralNetwork::new_load("models/model_1", &sigmoid, &cross_entropy).unwrap();

    // Run for 3 epochs
    for _ in 0..1 {
        println!(
            "{}",
            net_mse.train_multiple(0.06f64, train_data.clone()).unwrap()
        );
        println!(
            "{}",
            net_ce.train_multiple(0.06f64, train_data.clone()).unwrap()
        );
    }

    println!(
        "Diff between last prediction: MSE {0} vs CE {1}",
        net_mse.get_output(),
        net_ce.get_output()
    );

    // Save the model
    let _ = net_mse.save("models/net_mse");
    let _ = net_ce.save("models/net_ce");
}
