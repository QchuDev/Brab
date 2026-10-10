use nalgebra::DVector;
use brab::{
    activation::Sigmoid, 
    cost::Mse, 
    network::NeuralNetwork
};

fn main() {
    println!("Start..\n");

    let input: DVector<f64> = DVector::from_fn(4, |i , j| -> f64 { i as f64-j as f64});
    let expected: DVector<f64> = DVector::from_element(2, 0.9);
    
    println!("Sigmoid_NN created...");
    let mut net = NeuralNetwork::new(&Vec::from([4,5,2]), Sigmoid::new(), Mse::new());
    
    println!("Sigmoid_NN feeded...");

    for i in 1..1001 {
        net.feed_forward(input.clone());
        net.sgd(0.1f64, expected.clone());
        if i % 100 == 0 { println!("i = {i}\n{0}", net.get_output()); }
    }

    println!("\nStop..");
}

