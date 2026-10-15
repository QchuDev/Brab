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

    let _ = net.save("models/model_0");
    let mut net_1 = NeuralNetwork::new_load("models/model_0", Sigmoid::new(), Mse::new()).unwrap();
    
    let train_data = &mnist_prov.get_data()[0];

    // run it for some epochs and it will learn
    for _ in 0..1 {
        println!("{}", net.evaluate(train_data.input.clone()).unwrap());
        println!("{}", net_1.evaluate(train_data.input.clone()).unwrap());
    }
    
    let _ = net.save("models/model_1");

}


