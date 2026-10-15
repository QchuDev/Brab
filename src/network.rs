use core::f64;
use std::{error::Error, fs};

use crate::{activation, cost, data::{self, TrainData}};
use nalgebra::{self, DMatrix, DVector};
use rand::{random};
use safetensors::{Dtype, SafeTensorError, serialize, serialize_to_file, tensor::TensorView};
use tuple_conv::{RepeatedTuple, TupleOrVec};

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
                |_i, _j| -> f64 { random::<f64>() * 2.0 - 1.0 }
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
    
    /// Load an existing configuration from a (...).safetensor file
    pub fn new_load(file_path : &str) -> Self {
        //
        
    }


    /// Treat this neural network as a function:
    /// - f(x) -> y
    /// 
    /// It returns the output for a given input 
    pub fn evaluate(&mut self, input: DVector<f64>) -> Option<&DVector<f64>> {
        if let Ok(_msg) = self.feed_forward(input) {
            return Some(self.get_output());
        } else {
            return None
        }
    }

    pub fn train_multiple(&mut self, eta: f64, train_data: Vec<TrainData>) -> Result<&str,String> {
        let mut certainty = 0.0;
        let mut error = 0.0;
        let mut s_certainty = 0.0;
        let mut s_error = 0.0;
        let mut n = 0;
        let mut ok = 0;
        
        for td in train_data {
            let _ = self.feed_forward(td.input);
            
            if let Err(msg) = self.train_single(eta, td.expected.clone()) {
                return Err(msg.to_string());
            }

            n += 1;
            
            certainty = self.get_output().max();
            s_certainty += certainty;

            error = (self.get_output() - &td.expected).sum();
            s_error += error;

            if error < 0.02 && certainty > 0.8 { ok += 1 }
            println!("{0} vs {1}", td.expected, self.get_output());
            println!("Certainty  = {}", certainty);
            println!("Error  = {}", error);

        }
        println!("Mean Certainty  = {} \n Mean Error = {} \n Accuracy = {}", 
            s_certainty/n as f64, 
            s_error/n as f64,
            ok as f64/n as f64 
        );

        return Ok("All training data processed");
    }
    
    /// Call this method to calculate how well the network did with the aproximation 
    /// and update its parameters (weigths and biases)
    pub fn train_single(&mut self, eta: f64, expected: DVector<f64>) -> Result<&str, &str> {
        return self.sgd(eta, expected);
    }    

    pub fn get_output(&self) -> &DVector<f64> {
        self.layers[self.num_layers-1].get_a()
    }

    pub fn save(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        // Save all weights and biases into a CSV file in 'file_path'

        // First lets iterate for each weight and bias 
        for i in 0..self.num_layers-1 {
            // Transform both bias and weight into a chain of numbers (flatten)
            let vec_b: Vec<f64> = self.layers[i].get_b().as_slice().to_vec();
            let vec_w: Vec<f64> = self.weights[i].as_slice().to_vec();
            let w_shape = self.weights[i].shape().as_vec();
            
            // Translate into u8 values  
            let b_bytes: Vec<u8> = vec_b
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();
            let w_bytes: Vec<u8> = vec_w
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();

            // Create the tensor view for each one
            let b_tensor_view = TensorView::new(
                Dtype::F64,
                vec![vec_b.len()],
                &b_bytes,
            )?;

            let w_tensor_view = TensorView::new(
                Dtype::F64,
                w_shape,
                &w_bytes,
            )?;

            let file_data = vec![
                (format!["weights_{}", i], w_tensor_view),
                (format!["biases_{}", i], b_tensor_view)
            ];

            // Save into disk
            serialize_to_file(
                file_data,
                None,
                std::path::Path::new(file_path),
            )?;


        }

        // And lets not forget about that last handling bias vector
        let vec_b: Vec<f64> = self.layers[self.num_layers-1].get_b().as_slice().to_vec();
        // Translate into u8 values  
        let b_bytes: Vec<u8> = vec_b
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();

        // Create the tensor view for each one
        let b_tensor_view = TensorView::new(
            Dtype::F64,
            vec![vec_b.len()],
            &b_bytes,
        )?;

        let file_data = vec![
            (format!["biases_{}", self.num_layers-1], b_tensor_view)
        ];

        // Save into disk
        serialize_to_file(
            file_data,
            None,
            std::path::Path::new(file_path),
        )?;
        
        return Ok(());
    }

    // ==================== PRIV functions ==================== 

    /// Feed with an input to update the layers of the Neural Network
    fn feed_forward(&mut self, input: DVector<f64>) -> Result<&str, &str> {
        // Check if the input and first entry layer are the same size
        if input.nrows() != self.layers[0].size() { return Err("Not the same size"); }
        
        self.layers[0].set_a(&input);
        for i in 1..(self.num_layers) {
            let z_prime = &(&self.weights[i-1] * self.layers[i-1].get_a() + self.layers[i].get_b());
            self.layers[i].set_z(z_prime);  // we save the weigthed input for later
            
            let a_prime = &self.sigma(z_prime);
            self.layers[i].set_a(a_prime); 
        }

        return Ok("Feeded");

    }

    /// Apply "stocastic gradient descent"
    fn sgd(&mut self, eta: f64, expected: DVector<f64>) -> Result<&str, &str> {
        // First we run the back propagation algorithm to calculate the cost gradient
        let opt_gr = self.backprop(expected);
        if let None = opt_gr { return Err("ERROR: Cost Gradient couldn't be solved"); }
        
        let cost_gradient: CostGradient = opt_gr.unwrap();
        // We update the parameters of the network with the cost_gradient and eta (training rate)
        self.update_parameters(cost_gradient, eta);

        return Ok("Ok: Cost Gradient solved and parameters updated");
    }

    fn update_parameters(&mut self, cost_gradient: CostGradient, eta: f64) {
        for l in 0..self.num_layers-1 {
            let new_w = &self.weights[l] - eta * &cost_gradient.nabla_w[l];
            self.weights[l] = new_w;
    
            let new_b:DVector<f64> = self.layers[l+1].get_b() - eta * &cost_gradient.nabla_b[l];
            self.layers[l+1].set_b(&(new_b));
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
    z: DVector<f64>,      // weigthed inputs    z = b * a + b
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
                |_i, _j| -> f64 { random::<f64>() * 2.0 - 0.5 }
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

struct CostGradient {
    nabla_w: Vec<DMatrix<f64>>,
    nabla_b: Vec<DVector<f64>>
}