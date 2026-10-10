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
