use brab::{
    config::NetworkConfig,
    neural_functions::FunctionsRegistry,
    persistence::NetworkPersistence,
    training::{EpochResult, SGD, Trainer},
};
use nalgebra::DVector;

include!("mnist_data.rs");

fn main() {
    println!("Start..\n");
    for _ in 0..3 {
        let registry = FunctionsRegistry::default();
        let sigmoid = registry.get_activation("sigmoid").unwrap();
        let cross_entropy = registry.get_cost("cross entropy").unwrap();

        let model_path = "models/ce-sg-784-16-8-4-10";
        // if you want to load any other function...
        // registry.add_activation(...) or registry.add_cost(...);

        let _config = NetworkConfig::new()
            .with_shape(&[784, 16, 8, 4, 10])
            .with_activation(sigmoid)
            .with_cost(cross_entropy)
            .with_sofmax() // last layer uses softmax
            .build();

        // let mut network = NeuralNetwork::from_config(config);
        let mut network = NetworkPersistence::load(model_path, &registry).unwrap();
        let _ = NetworkPersistence::save(&mut network, model_path);

        let provider = MnistProvider::new();
        let train_data = provider.get_data();

        let results = Trainer::train(&mut network, train_data, &SGD::new(0.06f64), 1);

        print_results(results.unwrap());
    }
}

pub fn print_results(results: Vec<EpochResult>) {
    results
        .iter()
        .enumerate()
        .for_each(|(i, e_res)| println!("Results of epoch {}: \n{}", (i + 1), e_res));
}
