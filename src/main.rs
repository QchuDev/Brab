use brab::{
    config::NetworkConfig,
    consts::{CROSS_ENTROPY, SIGMOID},
    network::NeuralNetwork,
    training::{EpochResult, SGD, Trainer},
};
use nalgebra::DVector;

include!("mnist_data.rs");

fn main() {
    println!("Start..\n");

    let config = NetworkConfig::new()
        .with_shape(&[784, 16, 8, 4, 10])
        .with_activation(&SIGMOID)
        .with_cost(&CROSS_ENTROPY)
        .with_sofmax()
        .build();

    let mut network = NeuralNetwork::from_config(config);

    let provider = MnistProvider::new();
    let train_data = provider.get_data();

    let results = Trainer::train(&mut network, train_data, &SGD::new(0.06f64), 10);
    print_results(results.unwrap());
}

pub fn print_results(results: Vec<EpochResult>) {
    results
        .iter()
        .enumerate()
        .for_each(|(i, e_res)| println!("Results of epoch {}: \n{}", (i + 1), e_res));
}
