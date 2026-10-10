use nalgebra::{self, DMatrix, DVector};
use rand::{random};

/*
    On this NeuralNetwork we distinguish the hidden layers from the input and output ones
*/
pub struct NeuralNetwork<A, C> {
    layers: Vec<Layer>,                                 // layers of neurons: current activation values 
    weights: Vec<DMatrix<f64>>,                         // in-between weights matrices  
    num_layers: usize,                                  // number of layers 
    act_fn: A,          // activation function
    cst_fn: C,    // cost function
}
impl<A:ActivationFunction, C: CostFunction> NeuralNetwork<A, C> {
    
    /// Create a NEW Neural Network by passing size of each layer
    fn new(
        layers_sizes: &Vec<usize>, 
        new_act_fn: A,
        new_cst_fn: C
    ) -> Self {

        let new_num_layers = Vec::len(&layers_sizes);
        
        // Create each layer of zeros with the specified sizes
        let mut new_layers = Vec::new();
        for s in layers_sizes {
            new_layers.push(Layer::new(*s));
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

        return Self {
            layers:     new_layers,
            weights:    new_weights,
            num_layers: new_num_layers,
            act_fn: new_act_fn,
            cst_fn: new_cst_fn
        };
    }

    /// Feed with an input to update the layers of the Neural Network
    pub fn feed_forward(&mut self, input: DVector<f64>) -> () {
        // Check if the input and first entry layer are the same size
        if input.nrows() != self.layers[0].size() { return; }

        self.layers[0].set_a(&input);
        for i in 1..(self.num_layers) {
            let z_prime = &(&self.weights[i-1] * self.layers[i-1].get_a() + self.layers[i-1].get_b());
            self.layers[i].set_z(z_prime);  // we save the weigthed input for later
            
            let a_prime = &self.sigma(z_prime);
            self.layers[i].set_a(a_prime); 
        }
        
    }

    // Called to get the  partial derivatives 
    pub fn backprop(self) -> () {
        // Compute output_error

        // Backpropagate the error

        // output
    }
    
    fn output_error(self) -> DVector<f64> {
        let mut delta= DVector::zeros(self.layers[0].size());

        /// .... continue here
        delta
    }

    /// Compute the cost function with the expected vector.
    pub fn cost(&self, expected: &DVector<f64>) ->  f64 {
        self.cst_fn.function(expected, &self.get_output())
    }

    /// Calls the assigned activation functionx
    fn sigma(&self, x_ref: &DVector<f64>) -> DVector<f64> {
        self.act_fn.function(x_ref)
    }

    pub fn get_output(&self) -> DVector<f64> {
        self.layers[self.num_layers-1].get_a().clone()
    }

}

struct Layer {
    z: DVector<f64>,      // weigthed inputs    z = w * a + b
    a: DVector<f64>,      // activations : a = s(z)

    b: DVector<f64>       // biases
}
impl Layer {
    //TODO: missing size checking

    pub fn new(size:usize) -> Self{
        Self {
            a: DVector::zeros(size),
            z: DVector::zeros(size),
            
            b: DVector::from_fn(
                size,
                |_i, _j| -> f64 { random() }
            ),
        }
    }
    
    pub fn size(&self) -> usize { self.a.nrows() }

    // GETTERS AND SETTERS
    pub fn set_a(&mut self, new_a: &DVector<f64>) -> () { self.a = new_a.clone(); } 
    pub fn get_a(&self) -> &DVector<f64>{ &(self.a) } 

    pub fn set_z(&mut self, new_z: &DVector<f64>) -> () { self.z = new_z.clone(); } 
    pub fn get_z(&self) -> &DVector<f64>{ &(self.z) } 
    
    pub fn set_b(&mut self, new_b: &DVector<f64>) -> () { self.b = new_b.clone(); } // correct cloning??
    pub fn get_b(&self) -> &DVector<f64>{ &(self.b) } 
}