use crate::config::NetworkConfig;
use crate::consts::SOFTMAX;
use crate::neural_functions::NeuralFunctionContext;
use crate::neural_functions::{ActivationFunction, CostFunction};
use core::f64;
use nalgebra::{self, DMatrix, DVector};
use rand::random;

/*
    On this NeuralNetwork we distinguish the hidden layers from the input and output ones
*/
pub struct NeuralNetwork<'a> {
    layers: Vec<Layer<'a>>,       // layers of neurons: current activation values
    weights: Vec<DMatrix<f64>>,   // in-between weights matrices
    num_layers: usize,            // number of layers
    cst_fn: &'a dyn CostFunction, // cost function
}
impl<'a> NeuralNetwork<'a> {
    /// Create a NEW Neural Network by passing size of each layer
    pub fn from_config(config: NetworkConfig<'a>) -> Self {
        let new_num_layers = config.layers_sizes.len();

        // Create each layer of zeros with the specified sizes
        let mut new_layers = vec![];
        for (i, size) in config.layers_sizes.iter().enumerate() {
            if i == config.layers_sizes.len() - 1 && config.has_softmax {
                new_layers.push(Layer::new(*size, &SOFTMAX));
                break;
            }
            new_layers.push(Layer::new(*size, config.new_act_fn));
        }

        // Create each in-between matrix of random weights
        let mut new_weights: Vec<DMatrix<f64>> = vec![];
        for i in 0..(new_num_layers - 1) {
            new_weights.push(DMatrix::from_fn(
                config.layers_sizes[i + 1],
                config.layers_sizes[i],
                |_i, _j| -> f64 { random::<f64>() * 2.0 - 1.0 },
            ));
        }

        return Self {
            layers: new_layers,
            weights: new_weights,
            num_layers: new_num_layers,
            cst_fn: config.new_cst_fn,
        };
    }

    /// Feed with an input to update the layers of the Neural Network
    pub fn feed_forward(&mut self, input: DVector<f64>) -> Result<(), &str> {
        if input.len() != self.layers[0].get_a().len() {
            return Err("Input needs to be the same size as input layer");
        }

        self.layers[0].set_a(input);

        for i in 1..self.num_layers {
            let new_z = &self.weights[i - 1] * self.layers[i - 1].get_a() + self.layers[i].get_b();
            self.layers[i].set_z(new_z.clone()); // we save the weigthed input for later

            let new_a = self.layers[i].activation.fun(new_z);
            self.layers[i].set_a(new_a);
        }
        Ok(())
    }

    // Called to get the  partial derivatives
    pub fn backprop(&mut self, expected: DVector<f64>) -> Option<ParametersGradient> {
        // Vector to save tje errors for each layer
        let mut deltas: Vec<DVector<f64>> = Vec::new();
        for layer in &self.layers {
            // if (l == 0 || l == self.num_layers-1) { continue }  // needed???
            deltas.push(DVector::zeros(layer.size()));
        }

        // Compute output_error
        let context: NeuralFunctionContext = NeuralFunctionContext {
            a: &self.get_output(),
            y: &expected,
            z: &self.layers[self.num_layers - 1].z,
            activ_fn: self.layers[self.num_layers - 1].activation,
            cost_fn: self.cst_fn,
        };
        deltas[self.num_layers - 1] = self.cst_fn.delta(context);

        // Backpropagate the error
        for l in (1..self.num_layers).rev() {
            let p_error = &self.weights[l - 1].transpose() * &deltas[l];
            let sigmap_z = self.layers[l - 1]
                .activation
                .prime(self.layers[l - 1].get_z().clone());
            let error_l = p_error.component_mul(&sigmap_z); // <-- check this 

            deltas[l - 1] = error_l;
        }

        // output - Gradient of the cost function
        let mut nabla_b = Vec::new();
        let mut nabla_w = Vec::new();

        // iterate for each layer and use the errors to compute the partial derivative of each value
        for l in 1..self.num_layers {
            nabla_b.push(deltas[l].clone()); //  <<< hmmmm... check that
            nabla_w.push(&deltas[l] * &self.get_layer(l - 1).a.transpose());
        }

        return Some(ParametersGradient { nabla_w, nabla_b });
    }

    // GETTERS & SETTERS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    pub fn get_output(&self) -> &DVector<f64> {
        let ret = self.layers[self.num_layers - 1].get_a();
        ret
    }

    pub fn set_weight(&mut self, i: usize, new_w: DMatrix<f64>) {
        self.weights[i] = new_w;
    }

    pub fn get_weight(&mut self, i: usize) -> &DMatrix<f64> {
        &self.weights[i]
    }

    pub fn set_bias(&mut self, i: usize, new_b: DVector<f64>) {
        self.layers[i].set_b(new_b);
    }

    pub fn get_bias(&mut self, i: usize) -> &DVector<f64> {
        self.layers[i].get_b()
    }

    pub fn get_layer(&mut self, i: usize) -> &Layer<'_> {
        &self.layers[i]
    }

    /// Size of the Network is the ammount of layers, for weights subtract 1
    pub fn get_size(&self) -> usize {
        self.layers.len()
    }
}

/*
    Layer struct contains a, z and b column vectors. Also a reference to the activation function in use
*/
pub struct Layer<'a> {
    z: DVector<f64>, // weigthed inputs    z = b * a + b
    a: DVector<f64>, // activations : a = s(z)
    b: DVector<f64>, // biase
    activation: &'a dyn ActivationFunction,
}
impl<'a> Layer<'a> {
    pub fn new(size: usize, activation_function: &'a dyn ActivationFunction) -> Layer<'a> {
        Layer {
            a: DVector::zeros(size),
            z: DVector::zeros(size),

            b: DVector::from_fn(size, |_i, _j| -> f64 { random::<f64>() * 2.0 - 0.5 }),
            activation: activation_function,
        }
    }

    pub fn size(&self) -> usize {
        self.a.nrows()
    }

    pub fn get_a(&self) -> &DVector<f64> {
        &self.a
    }
    pub fn set_a(&mut self, new_a: DVector<f64>) {
        self.a = new_a
    }
    pub fn get_z(&self) -> &DVector<f64> {
        &self.z
    }
    pub fn set_z(&mut self, new_z: DVector<f64>) {
        self.z = new_z
    }
    pub fn get_b(&self) -> &DVector<f64> {
        &self.b
    }
    pub fn set_b(&mut self, new_b: DVector<f64>) {
        self.b = new_b
    }
}

/// Weight and biases gradient got from backpropagation
pub struct ParametersGradient {
    pub nabla_w: Vec<DMatrix<f64>>,
    pub nabla_b: Vec<DVector<f64>>,
}
