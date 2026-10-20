use crate::neural_functions::{CrossEntropy, Sigmoid, SoftMax};

pub static SOFTMAX: SoftMax = SoftMax::new();
pub static CROSS_ENTROPY: CrossEntropy = CrossEntropy::new();
pub static SIGMOID: Sigmoid = Sigmoid::new();
