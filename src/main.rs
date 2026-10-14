use core::f64;

use nalgebra::DVector;
use brab::{
    activation::Sigmoid, 
    cost::Mse, 
    network::NeuralNetwork
};

include!("mnist_data.rs");

fn main() {
    println!("Start..\n");
    let mut net = NeuralNetwork::new(&Vec::from([784, 10, 10, 10]), Sigmoid::new(), Mse::new());

    let mnist_prov = MnistProvider::new();
    
    // run it for some epochs and it will learn
    for _ in 0..10 {
        println!("{}", net.train_multiple(0.06f64, mnist_prov.get_data()).unwrap());
    }

}