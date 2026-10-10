include!("neural_network.rs");
include!("activation_functions.rs");
include!("cost_functions.rs");

/**
 * IMPORTANT CHECK FOR THE ERROR ON THE ADDTION AT FEEDFORWARD
 */

fn main() {

    println!("Start..\n");
    // Create the neural network by specifying: LAYERS SIZES and ACTIVATION FUNCTION     
    let mut relu_nn = NeuralNetwork::new(&Vec::from([2,5,2]), Relu::new(), MSE::new());
    println!("Relu_NN initialized...");

    // FEED it TO UPDATE the layers values
    let input = DVector::from_element(2, -20.0);

    relu_nn.feed_forward(input.clone());
    println!("Relu_NN feeded...");
    println!("{0}", relu_nn.get_output());

    
    let mut sigmoid_nn = NeuralNetwork::new(&Vec::from([2,5,2]), Sigmoid::new(), MSE::new());
    sigmoid_nn.feed_forward(input);
    println!("{0}", sigmoid_nn.get_output());
    
    println!("\nStop..");
    
}

