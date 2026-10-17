use crate::neural_functions::{ActivationFunction, CostFunction, SoftMax};
use crate::{data::TrainData, neural_functions::NeuralFunctionContext};
use core::f64;
use nalgebra::{self, DMatrix, DVector};
use rand::random;
use safetensors::{Dtype, SafeTensorError, SafeTensors, serialize_to_file, tensor::TensorView};
use std::error::Error;
use tuple_conv::TupleOrVec;

static SOFTMAX: SoftMax = SoftMax::new();
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
    pub fn new(config: NetworkConfig<'a>) -> NeuralNetwork<'a> {
        let new_num_layers = Vec::len(&config.layers_sizes);

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

        return NeuralNetwork {
            layers: new_layers,
            weights: new_weights,
            num_layers: new_num_layers,
            cst_fn: config.new_cst_fn,
        };
    }

    /// Load an existing configuration from a '(...).safetensor' file
    pub fn new_load(
        file_path: &str,
        act_fn: &'a dyn ActivationFunction,
        cst_fn: &'a dyn CostFunction,
        has_softmax: bool,
    ) -> Result<NeuralNetwork<'a>, SafeTensorError> {
        let file = std::fs::read(file_path)?;
        let tensors = SafeTensors::deserialize(&file)?;

        let num_layers = tensors
            .names()
            .iter()
            .filter(|n| n.starts_with("biases_"))
            .count();

        let mut new_layers: Vec<Layer> = Vec::with_capacity(num_layers);
        let mut new_weights: Vec<DMatrix<f64>> = Vec::with_capacity(num_layers - 1);

        // We know its indexed like 'name_{i}' so we use it to bring the values

        // Extracting biases
        for l in 0..num_layers {
            let bias_view = tensors.tensor(&format!("biases_{}", l))?;
            let (bias_data, _shape) = tensor_to_f64(&bias_view);

            let b = DVector::from_vec(bias_data);
            let size = b.nrows();

            if has_softmax && l == num_layers - 1 {
                new_layers.push(Layer {
                    a: DVector::zeros(size),
                    z: DVector::zeros(size),
                    b: b,
                    activation: &SOFTMAX,
                });
                break;
            }

            new_layers.push(Layer {
                a: DVector::zeros(size),
                z: DVector::zeros(size),
                b: b,
                activation: act_fn,
            });
        }

        // Extracting weights
        for l in 0..(num_layers - 1) {
            // We know its indexed like 'name_{i}' so we use it to bring the values
            let w_view = tensors.tensor(&format!("weights_{}", l))?;
            let (w_data, w_shape) = tensor_to_f64(&w_view);

            let rows = w_shape[0];
            let cols = w_shape[1];

            let w = DMatrix::from_column_slice(rows, cols, &w_data);
            new_weights.push(w);
        }

        Ok(Self {
            layers: new_layers,
            weights: new_weights,
            num_layers,
            cst_fn,
        })
    }

    /// Treat this neural network as a function:
    /// - f(x) -> y
    ///
    /// It returns the output for a given input
    pub fn evaluate(&mut self, input: DVector<f64>) -> Option<&DVector<f64>> {
        if let Ok(_msg) = self.feed_forward(input) {
            return Some(self.get_output());
        } else {
            return None;
        }
    }

    pub fn train_multiple(&mut self, eta: f64, train_data: Vec<TrainData>) -> Result<&str, String> {
        for td in train_data {
            let _ = self.feed_forward(td.input);

            if let Err(msg) = self.sgd(eta, td.expected.clone()) {
                return Err(msg.to_string());
            }
        }

        return Ok("All training data processed");
    }

    /// Call this method to calculate how well the network did with the aproximation
    /// and update its parameters (weigths and biases)
    pub fn train_single(&mut self, eta: f64, td: TrainData) -> Result<&str, &str> {
        let _ = self.feed_forward(td.input);
        return self.sgd(eta, td.expected);
    }

    pub fn get_output(&self) -> &DVector<f64> {
        self.layers[self.num_layers - 1].get_a()
    }

    pub fn save(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        // Save all weights and biases into a safetensors file in 'file_path'.
        //
        // IMPORTANT: `serialize_to_file` overwrites the whole file on every call,
        // so every tensor must be serialized in a SINGLE call. We first build all
        // the byte buffers (which must outlive the TensorViews that borrow them),
        // then create the views, then serialize once.

        // 1) Flatten every bias and weight into little-endian byte buffers.
        //    We keep these buffers alive in `buffers` for the duration of the fn.
        let mut b_buffers: Vec<Vec<u8>> = Vec::with_capacity(self.num_layers);
        let mut w_buffers: Vec<Vec<u8>> = Vec::with_capacity(self.num_layers - 1);
        let mut b_sizes: Vec<usize> = Vec::with_capacity(self.num_layers);
        let mut w_shapes: Vec<Vec<usize>> = Vec::with_capacity(self.num_layers - 1);

        // All biases (one per layer, including input and output layers)
        for l in 0..self.num_layers {
            let vec_b: Vec<f64> = self.layers[l].get_b().as_slice().to_vec();
            b_sizes.push(vec_b.len());
            b_buffers.push(vec_b.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        // All weight matrices (num_layers - 1 of them)
        for l in 0..self.num_layers - 1 {
            let vec_w: Vec<f64> = self.weights[l].as_slice().to_vec();
            w_shapes.push(self.weights[l].shape().as_vec());
            w_buffers.push(vec_w.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        // 2) Build all tensor views, borrowing from the buffers above.
        let mut file_data: Vec<(String, TensorView)> = Vec::new();

        for l in 0..self.num_layers {
            let view = TensorView::new(Dtype::F64, vec![b_sizes[l]], &b_buffers[l])?;
            file_data.push((format!("biases_{}", l), view));
        }

        for l in 0..self.num_layers - 1 {
            let view = TensorView::new(Dtype::F64, w_shapes[l].clone(), &w_buffers[l])?;
            file_data.push((format!("weights_{}", l), view));
        }

        // 3) Serialize everything in one go.
        serialize_to_file(file_data, None, std::path::Path::new(file_path))?;

        return Ok(());
    }

    // ==================== PRIV functions ====================

    /// Feed with an input to update the layers of the Neural Network
    fn feed_forward(&mut self, input: DVector<f64>) -> Result<&str, &str> {
        // Check if the input and first entry layer are the same size
        if input.nrows() != self.layers[0].size() {
            return Err("Not the same size");
        }

        // input case (just set the input)
        self.layers[0].set_a(&input);

        // hidden layers
        // remember to change this for n-1 when hidden layers
        // only
        for i in 1..(self.num_layers) {
            let new_z =
                &(&self.weights[i - 1] * self.layers[i - 1].get_a() + self.layers[i].get_b());
            self.layers[i].set_z(new_z); // we save the weigthed input for later

            let new_a = &self.layers[i].activation.fun(new_z);
            self.layers[i].set_a(new_a);
        }

        return Ok("Yummy! Nice input");
    }

    /// Apply "stocastic gradient descent"
    fn sgd(&mut self, eta: f64, expected: DVector<f64>) -> Result<&str, &str> {
        // First we run the back propagation algorithm to calculate the cost gradient
        let opt_gr = self.backprop(expected);
        if let None = opt_gr {
            return Err("ERROR: Cost Gradient couldn't be solved");
        }

        let cost_gradient: CostGradient = opt_gr.unwrap();
        // We update the parameters of the network with the cost_gradient and eta (training rate)
        self.update_parameters(cost_gradient, eta);

        return Ok("Ok: Cost Gradient solved and parameters updated");
    }

    fn update_parameters(&mut self, cost_gradient: CostGradient, eta: f64) {
        for l in 0..self.num_layers - 1 {
            let new_w = &self.weights[l] - eta * &cost_gradient.nabla_w[l];
            self.weights[l] = new_w;

            let new_b: DVector<f64> = self.layers[l + 1].get_b() - eta * &cost_gradient.nabla_b[l];
            self.layers[l + 1].set_b(&(new_b));
        }
    }

    // Called to get the  partial derivatives
    fn backprop(&mut self, expected: DVector<f64>) -> Option<CostGradient> {
        // Vector to save tje errors for each layer
        let mut deltas: Vec<DVector<f64>> = Vec::new();
        for layer in &self.layers {
            // if (l == 0 || l == self.num_layers-1) { continue }  // needed???
            deltas.push(DVector::zeros(layer.size()));
        }

        // Compute output_error
        let context: NeuralFunctionContext = NeuralFunctionContext {
            a: self.get_output(),
            y: &expected,
            z: self.layers[self.num_layers - 1].get_z(),
            activ_fn: self.layers[self.num_layers - 1].activation,
            cost_fn: self.cst_fn,
        };
        let output_error = self.cst_fn.delta(context);
        deltas[self.num_layers - 1] = output_error;

        // Backpropagate the error
        for l in (1..self.num_layers).rev() {
            let p_error = &self.weights[l - 1].transpose() * &deltas[l];
            let sigmap_z = self.layers[l - 1]
                .activation
                .prime(self.layers[l - 1].get_z());
            let error_l = p_error.component_mul(&sigmap_z); // <-- check this 

            deltas[l - 1] = error_l;
        }

        // output - Gradient of the cost function
        let mut nabla_b = Vec::new();
        let mut nabla_w = Vec::new();

        // iterate for each layer and use the errors to compute the partial derivative of each value
        for l in 1..self.num_layers {
            nabla_b.push(deltas[l].clone()); //  <<< hmmmm... check that
            nabla_w.push(&deltas[l] * &self.layers[l - 1].get_a().transpose());
        }

        return Some(CostGradient { nabla_w, nabla_b });
    }
}

pub struct NetworkConfig<'a> {
    pub layers_sizes: &'a Vec<usize>,
    pub new_act_fn: &'a dyn ActivationFunction,
    pub new_cst_fn: &'a dyn CostFunction,
    pub has_softmax: bool,
}

struct Layer<'a> {
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

    // GETTERS AND SETTERS
    pub fn set_a(&mut self, new_a: &DVector<f64>) -> () {
        self.a = new_a.clone();
    }
    pub fn get_a(&self) -> &DVector<f64> {
        &(self.a)
    }

    pub fn set_z(&mut self, new_z: &DVector<f64>) -> () {
        self.z = new_z.clone();
    }
    pub fn get_z(&self) -> &DVector<f64> {
        &(self.z)
    }

    pub fn set_b(&mut self, new_b: &DVector<f64>) -> () {
        self.b = new_b.clone();
    } // correct cloning??
    pub fn get_b(&self) -> &DVector<f64> {
        &(self.b)
    }
}

struct CostGradient {
    nabla_w: Vec<DMatrix<f64>>,
    nabla_b: Vec<DVector<f64>>,
}

fn tensor_to_f64(view: &TensorView) -> (Vec<f64>, Vec<usize>) {
    let data: Vec<f64> = view
        .data()
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (data, view.shape().to_vec())
}
