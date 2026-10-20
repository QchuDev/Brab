use std::error::Error;

use nalgebra::{DMatrix, DVector};
use safetensors::{Dtype, SafeTensorError, SafeTensors, serialize_to_file, tensor::TensorView};
use tuple_conv::TupleOrVec;

use crate::{
    config::NetworkConfig,
    consts::SOFTMAX,
    network::{Layer, NeuralNetwork},
    neural_functions::{ActivationFunction, CostFunction},
};

// TODO: Rework the persistence, save more stuff to load better

/// Save & Loader of models
pub struct NetworkPersistence {}
impl NetworkPersistence {
    pub fn save(network: &mut NeuralNetwork, file_path: &str) -> Result<String, Box<dyn Error>> {
        // so every tensor must be serialized in a SINGLE call. We first build all
        // the byte buffers (which must outlive the TensorViews that borrow them),
        // then create the views, then serialize once.

        // 1) Flatten every bias and weight into little-endian byte buffers.
        //    We keep these buffers alive in `buffers` for the duration of the fn.
        let mut b_buffers: Vec<Vec<u8>> = Vec::with_capacity(network.get_size());
        let mut w_buffers: Vec<Vec<u8>> = Vec::with_capacity(network.get_size() - 1);
        let mut b_sizes: Vec<usize> = Vec::with_capacity(network.get_size());
        let mut w_shapes: Vec<Vec<usize>> = Vec::with_capacity(network.get_size() - 1);

        // All biases (one per layer, including input and output layers)
        for l in 0..network.get_size() {
            let vec_b: Vec<f64> = network.get_bias(l).as_slice().to_vec();
            b_sizes.push(vec_b.len());
            b_buffers.push(vec_b.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        // All weight matrices (num_layers - 1 of them)
        for l in 0..network.get_size() - 1 {
            let vec_w: Vec<f64> = network.get_weight(l).as_slice().to_vec();
            w_shapes.push(network.get_weight(l).shape().as_vec());
            w_buffers.push(vec_w.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        // 2) Build all tensor views, borrowing from the buffers above.
        let mut file_data: Vec<(String, TensorView)> = Vec::new();

        for l in 0..network.get_size() {
            let view = TensorView::new(Dtype::F64, vec![b_sizes[l]], &b_buffers[l])?;
            file_data.push((format!("biases_{}", l), view));
        }

        for l in 0..network.get_size() - 1 {
            let view = TensorView::new(Dtype::F64, w_shapes[l].clone(), &w_buffers[l])?;
            file_data.push((format!("weights_{}", l), view));
        }

        // 3) Serialize everything in one go.
        serialize_to_file(file_data, None, std::path::Path::new(file_path))?;

        return Ok("Neural Network correctly saved".to_string());
    }

    pub fn load<'a>(
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

            let mut nl = Layer::new(size, act_fn);
            // last layer adding softmax
            if has_softmax && l == num_layers - 1 {
                nl = Layer::new(size, &SOFTMAX);
            }

            // setting bias into layer
            new_layers.push(nl);
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

        // we create the default network for now
        let config = NetworkConfig::new()
            .with_activation(act_fn)
            .with_cost(cst_fn)
            .build();

        let mut net = NeuralNetwork::from_config(config);
        for i in 0..num_layers {
            net.set_bias(i, new_layers[i].get_b().clone());
            if i < num_layers - 1 {
                net.set_weight(i, new_weights[i].clone());
            }
        }

        Ok(net)
    }
}

fn tensor_to_f64(view: &TensorView) -> (Vec<f64>, Vec<usize>) {
    let data: Vec<f64> = view
        .data()
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (data, view.shape().to_vec())
}
