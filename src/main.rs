include!("neural_network.rs");
include!("activation_functions.rs");
include!("cost_functions.rs");

/**
 * IMPORTANT CHECK FOR THE ERROR ON THE ADDTION AT FEEDFORWARD
 */

fn main() {
    println!("Start..\n");

    
    let input: DVector<f64> = DVector::from_element(2, -20.0);
    let expected: DVector<f64> = DVector::from_element(2, 0.0);


    // println!("Relu_NN created...");
    // let mut relu_nn = NeuralNetwork::new(&Vec::from([2,5,2]), Relu::new(), Mse::new());
    // relu_nn.feed_forward(input.clone());
    // println!("Relu_NN feeded...");
    // println!("{0}", relu_nn.get_output());


    
    println!("Sigmoid_NN created...");
    let mut sigmoid_nn = NeuralNetwork::new(&Vec::from([2,5,2]), Sigmoid::new(), Mse::new());

    sigmoid_nn.feed_forward(input);
    println!("Sigmoid_NN feeded...");

    if let Some(gr) = sigmoid_nn.backprop(expected) {
        println!("Cost Gradient weights: {:?}", gr.nabla_w); 
        println!("Cost Gradient biases: {:?}", gr.nabla_b); 
    } 
    else { 
        println!("No COST GRADIENT")
    }


    println!("Sigmoid_NN backprop...");

    println!("{0}", sigmoid_nn.get_output());
    


    println!("\nStop..");
}

