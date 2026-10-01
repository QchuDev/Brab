use nalgebra::{self, DMatrix, DVector};
use rand::{random};

/*
    On this NeuralNetwork we distinguish the hidden layers from the input and output ones
*/
pub struct NeuralNetwork {
    layers: Vec<DVector<f64>>,  // layers of neurons: current activation values 
    weights: Vec<DMatrix<f64>>, // in-between weights matrices  
    biases: Vec<DVector<f64>>,  // biases for each hidden layer
    num_layers: usize,          // number of layers 
    act_fn: fn(&DVector<f64>) -> DVector<f64>
}
impl NeuralNetwork {
    
    /// Create a NEW Neural Network by passing size of each layer
    fn new(layers_sizes: &Vec<usize>, new_act_fn: fn(&DVector<f64>) -> DVector<f64>) -> Self {

        let new_num_layers = Vec::len(&layers_sizes);
        
        // Create each layer of zeros with the specified sizes
        let mut new_layers = Vec::new();
        for s in layers_sizes {
            new_layers.push(DVector::zeros(*s));
        }

        // Create each in-between matrix of random weights
        let mut new_weights:Vec<DMatrix<f64>> = Vec::new();
        for i  in 0..(new_num_layers-1) {
            new_weights.push(DMatrix::from_fn(
                layers_sizes[i+1],
                layers_sizes[i],
                |_i, _j| -> f64 { random() }
            ));
        }

        // Create hidden layer's biases
        let mut new_biases = Vec::new();
        for i  in 1..(new_num_layers) {
            new_biases.push(DVector::from_fn(
                layers_sizes[i],
                |_i, _j| -> f64 { random() }
            ));
        }

        return Self {
            layers:     new_layers,
            weights:    new_weights,
            biases:     new_biases,
            num_layers: new_num_layers,
            act_fn: new_act_fn
        };
    }

    /// Feed with an input to update the layers of the Neural Network
    pub fn feed(&mut self, input: DVector<f64>) -> () {
        // Check if the input and first entry layer are the same size
        if input.nrows() != self.layers[0].nrows() { return; }

        self.layers[0] = input;
        for i in 1..(self.num_layers) {
            self.layers[i] = self._act_fn(&(&self.weights[i-1] * &self.layers[i-1] + &self.biases[i-1]));
        }
        
    }
    /// Calls the assigned activation functionx
    fn _act_fn(&self, x_ref: &DVector<f64>) -> DVector<f64> {
        (self.act_fn)(x_ref)
    }


    /// Get the ouput layer
    pub fn get_output(&self) -> DVector<f64> {
        self.layers[self.num_layers-1].clone()
    }

}

/*  
================        ACTIVATION FUNCTIONS        ================
*/
fn relu(x: &DVector<f64>) -> DVector<f64>{
    let mut ret = DVector::from(x.clone());
    for i in 0..DVector::nrows(&ret) {
        if ret[i] < 0.0 {
            ret[i] = 0.0;
        }

    }
    return ret;
}