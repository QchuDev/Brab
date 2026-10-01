include!("structure.rs");

fn main() {

    println!("Start..\n");

    // Create the neural network by specifying: LAYERS SIZES and ACTIVATION FUNCTION     
    let mut nn = NeuralNetwork::new(&Vec::from([2,5,2]), relu);

    // FEED it TO UPDATE the layers values
    let input = DVector::from_element(2, -20.0);
    nn.feed(input);
    println!("{0}", nn.get_output());
    
    // CALCULATE the error from the prediction 
    // .. MSE

    // CORRECT the weight values
    // ... back propagation

    println!("\nStop..");
    
}

