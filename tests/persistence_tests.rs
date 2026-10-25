use brab::{
    config::NetworkConfig, network::NeuralNetwork, neural_functions::FunctionsRegistry,
    persistence::NetworkPersistence,
};

#[test]
fn tets_save_and_load() {
    // how do I tests this...? simulate the use of safetensors is possible? is really necessary to
    // do it so abstract or mock it ?

    let registry = FunctionsRegistry::default();
    let mut network = NeuralNetwork::from_config(NetworkConfig::new().with_shape(&[2, 4, 2]));
    let save_result = NetworkPersistence::save(&mut network, "tests/persistence_env/test_model");
    let load_result = NetworkPersistence::load("tests/persistence_env/test_model", &registry);

    assert!(save_result.is_ok());
    assert!(load_result.is_ok());

    let mut loaded_net = load_result.unwrap();
    assert!(loaded_net.get_size() == network.get_size());
    let n = network.get_size();
    for i in 0..n {
        assert!(loaded_net.get_bias(i).eq(network.get_bias(i)));
    }
    for i in 0..n - 1 {
        assert!(loaded_net.get_weight(i).eq(network.get_weight(i)));
    }
}
