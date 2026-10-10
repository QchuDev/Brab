pub mod activation {
    use nalgebra::DVector;

    
    /*  
    ================        ACTIVATION FUNCTIONS        ================
    */
    /// This traits expects an implementation for:
    /// Activation function (function)
    /// and its derivative (derivative)
    pub trait ActivationFunction {
        fn function(&self, x: &DVector<f64>) -> DVector<f64>;
        fn derivative(&self, x: &DVector<f64>) -> DVector<f64>;
    }

    pub struct Relu{}
    impl Relu { pub fn new() -> Self {  Self{} }}
    impl ActivationFunction for Relu {

        fn function(&self, x: &DVector<f64>) -> DVector<f64> {
            let mut ret = DVector::from(x.clone());
            for i in 0..DVector::nrows(&ret) {
                ret[i] = relu_f(ret[i]);
            }
            return ret;
        }
        fn derivative(&self, x: &DVector<f64>) -> DVector<f64> { 
            let mut ret = DVector::from(x.clone());
            for i in 0..DVector::nrows(&ret) {
                
                if ret[i] > 0.0  {  ret[i] = 1.0;}
                else { ret[i] = 0.0 }
            }
            return ret;
        }
    }
    fn relu_f(x: f64) -> f64 { if x < 0.0 { 0.0 } else { x } }


    pub struct Sigmoid{}
    impl Sigmoid { pub fn new() -> Self {  Self{} }}
    impl ActivationFunction for Sigmoid  {

        fn function(&self, x: &DVector<f64>) -> DVector<f64> {
            let mut ret = DVector::from(x.clone());
            for i in 0..DVector::nrows(&ret) {
                ret[i] = sigmoid_f(ret[i]);
            }
            return ret;
        }
        fn derivative(&self, x: &DVector<f64>) -> DVector<f64> {
            let mut ret = DVector::from(x.clone());
            for i in 0..DVector::nrows(&ret) {
                ret[i] = sigmoid_prime_f(ret[i]);
            }
            return ret;
        }
    }
    fn sigmoid_f(x: f64) -> f64 { 1f64 / (1f64 + (-x).exp())  }
    fn sigmoid_prime_f(x: f64) -> f64 { sigmoid_f(x) * (1f64 - sigmoid_f(x))  }
}

pub mod cost {
    use nalgebra::DVector;

    /// This CostFunction trait expects you to implement: the function itself and its gradient respect to a (activation)
    pub trait CostFunction {
        fn function(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64;
        fn gradient_a(&self, expected: &DVector<f64>, response: &DVector<f64>) -> Option<DVector<f64>>;
        
    }

    /*  
    ================        COST FUNCTIONS        ================
    */
    pub struct Mse {}
    impl Mse { pub fn new() -> Self {  Self{} }}
    impl CostFunction for Mse {
        fn function(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64 { mse(expected, response) }
        fn gradient_a(&self, expected: &DVector<f64>, response: &DVector<f64>) -> Option<DVector<f64>> { mse_gradient_a(expected, response)}
    }
    pub fn mse(expected: &DVector<f64>, response: &DVector<f64>) -> f64 {
        if expected.nrows() != response.nrows() { return 0f64; }
        
        let n: usize = expected.nrows();
        let mut sum: f64 = 0.0;

        for i in 0..n {
            sum += ((expected[i] - response[i])).powi(2);
        }

        return sum / (n as f64) ;
    }
    pub fn mse_gradient_a(expected: &DVector<f64>, response: &DVector<f64>) -> Option<DVector<f64>> {
        if expected.nrows() != response.nrows() { return None; }
        
        return Some(expected - response);
    }
}

pub mod network {
    use nalgebra::{self, DMatrix, DVector};
    use rand::{random};

use crate::brab::{activation, cost};

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
    impl<A:activation::ActivationFunction, C: cost::CostFunction> NeuralNetwork<A, C> {
        
        /// Create a NEW Neural Network by passing size of each layer
        pub fn new(
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
                let z_prime = &(&self.weights[i-1] * self.layers[i-1].get_a() + self.layers[i].get_b());
                self.layers[i].set_z(z_prime);  // we save the weigthed input for later
                
                let a_prime = &self.sigma(z_prime);
                self.layers[i].set_a(a_prime); 
            }
            
        }

        // Called to get the  partial derivatives 
        fn backprop(&mut self, expected: DVector<f64>) -> Option<CostGradient> {
            // Vector to save tje errors for each layer
            let mut errors:Vec<DVector<f64>> = Vec::new();
            for layer in &self.layers {
                // if (l == 0 || l == self.num_layers-1) { continue }  // needed???
                errors.push(DVector::zeros(layer.size()));
            }

            
            // Compute output_error
            let opt_oe = self.output_error(&expected, self.layers[self.num_layers-1].get_z());
            if let Some(output_error) = opt_oe {
                errors[self.num_layers-1] = output_error;  
            }  else {
                return None; 
            }


            // Backpropagate the error
            for l in (1..self.num_layers).rev() {
                let p_error = &self.weights[l-1].transpose() * &errors[l];
                let sigmap_z = &self.act_fn.derivative(self.layers[l-1].get_z());

                let error_l = p_error.component_mul(sigmap_z); // <-- check this 
                
                errors[l-1] = error_l;
            }

            // output - Gradient of the cost function
            let mut nabla_b = Vec::new();
            let mut nabla_w = Vec::new();

            // iterate for each layer and use the errors to compute the partial derivative of each value 
            for l in 1..self.num_layers {
                nabla_b.push(errors[l].clone());                                     //  <<< hmmmm... check that
                nabla_w.push(&errors[l] * &self.layers[l-1].get_a().transpose());
            }


            return Some(CostGradient{nabla_w, nabla_b});
        }

        /// Apply stocastic gradient descent
        pub fn sgd(&mut self, eta: f64, expected: DVector<f64>) -> () {
            let opt_gr = self.backprop(expected);

            if let Some(cost_gradient) = opt_gr {

                for l in 0..self.num_layers-1 {
                    let new_w = &self.weights[l] + eta * &cost_gradient.nabla_w[l];
                    self.weights[l] = new_w;

                    let new_b:DVector<f64> = self.layers[l+1].get_b() + eta * &cost_gradient.nabla_b[l];
                    self.layers[l+1].set_b(&(new_b));
                }
            } else {
                println!("NO COST FUNCTION GRADIENT");
            }
        }

        pub fn get_output(&self) -> &DVector<f64> {
            self.layers[self.num_layers-1].get_a()
        }
        
        /// Compute the cost function with the expected vector.
        pub fn cost(&self, expected: &DVector<f64>) ->  f64 {
            self.cst_fn.function(expected, &self.get_output())
        }



        // ==================== AUX functions ==================== 

        /// Measuring the error of the output layer    
        fn output_error(&self, expected: &DVector<f64>, z: &DVector<f64>) -> Option<DVector<f64>> {
            // Gradient respect to a
            let option_cgra = self.cst_fn.gradient_a(expected, self.get_output());
            if let Some(gradient_a) = option_cgra {
                Some(gradient_a.component_mul(&self.act_fn.derivative(z)))
            } else {
                None
            }
        }

        /// Calls the assigned activation functionx
        fn sigma(&self, x_ref: &DVector<f64>) -> DVector<f64> {
            self.act_fn.function(x_ref)
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

    pub struct CostGradient {
        nabla_w: Vec<DMatrix<f64>>,
        nabla_b: Vec<DVector<f64>>
    }
}
