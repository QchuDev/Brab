use brab::{
    config::NetworkConfig,
    network::{Layer, NeuralNetwork},
    neural_functions::{ActivationFunction, CostFunction, NeuralFunctionContext},
};
use nalgebra::DVector;

// yeah... there are no tests by now
//
// is that bad? nah, for now its not, I trust my code.
// Of course it will fail, then I'll debug and test specific types
struct DummyActivation {}
impl ActivationFunction for DummyActivation {
    fn fun(&self, x: DVector<f64>) -> DVector<f64> {
        x.clone()
    }
    fn prime(&self, x: DVector<f64>) -> DVector<f64> {
        x.clone()
    }
    fn name(&self) -> String {
        "dummy".to_string()
    }
}

struct DummyCost {}
impl CostFunction for DummyCost {
    fn delta(&self, context: NeuralFunctionContext) -> DVector<f64> {
        DVector::zeros(context.a.len())
    }
    fn name(&self) -> String {
        "dummy".to_string()
    }
}

#[test]
fn test_layers() {
    let dim = 3;
    let mut layer = Layer::new(dim, &DummyActivation {});

    layer.set_a(DVector::repeat(3, 1.0));
    layer.set_z(DVector::repeat(3, 1.0));
    layer.set_b(DVector::repeat(3, 1.0));

    assert!(layer.get_a().eq(&DVector::repeat(3, 1.0)));
    assert!(layer.get_z().eq(&DVector::repeat(3, 1.0)));
    assert!(layer.get_b().eq(&DVector::repeat(3, 1.0)));
}

#[test]
fn test_network_feed() {
    let config = NetworkConfig::new()
        .with_shape(&[3, 4, 4])
        .with_activation(&DummyActivation {})
        .with_cost(&DummyCost {})
        .with_sofmax();

    let mut net = NeuralNetwork::from_config(config);
    let input = DVector::from(vec![1.0, 1.0, 1.0]);

    assert!(net.feed_forward(input.clone()).is_ok());
}

#[test]
fn test_network_backprop() {}
