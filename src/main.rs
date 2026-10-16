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

    let net_shape: Vec<usize> = vec![784, 10, 10, 10];

    // Create network
    let mut net_mse: NeuralNetwork = NeuralNetwork::new(&net_shape, &sigmoid, &mse);
    let mut net_ce: NeuralNetwork = NeuralNetwork::new(&net_shape, &sigmoid, &cross_entropy);

    // Run for 3 epochs
    for _ in 0..15 {
        println!(
            "{}",
            net_mse.train_multiple(0.06f64, train_data.clone()).unwrap()
        );
        println!(
            "{}",
            net_ce.train_multiple(0.06f64, train_data.clone()).unwrap()
        );

        // Diff between epoch
        println!(
            "Diff between the max: MSE ({0}) vs CE ({1})",
            net_mse.get_output().max(),
            net_ce.get_output().max()
        );
    }

    // Save the model
    let _ = net_mse.save("models/net_mse");
    let _ = net_ce.save("models/net_ce");
}
