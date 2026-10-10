use nalgebra::DVector;

mod brab;
use crate::brab::network::NeuralNetwork;
use crate::brab::cost::Mse;
use crate::brab::activation::Sigmoid;

/**
 * IMPORTANT CHECK FOR THE ERROR ON THE ADDTION AT FEEDFORWARD
 */

fn main() {
    println!("Start..\n");

    let input: DVector<f64> = DVector::from_element(2, 5.0);
    let expected: DVector<f64> = DVector::from_element(3, 0.789);
    
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

