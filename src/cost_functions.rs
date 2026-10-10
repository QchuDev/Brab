pub trait CostFunction {
    fn function(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64;
    fn derivative(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64;
}

/*  
================        COST FUNCTIONS        ================
*/
struct MSE {}
impl MSE { pub fn new() -> Self {  Self{} }}
impl CostFunction for MSE{
    fn function(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64 { mse(expected, response) }
    fn derivative(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64 { mse_prime(expected, response)}
  
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
pub fn mse_prime(expected: &DVector<f64>, response: &DVector<f64>) -> f64 {
    if expected.nrows() != response.nrows() { return 0f64; }
    
    let n: usize = expected.nrows();
    let mut sum: f64 = 0.0;

    for i in 0..n {
        sum += (expected[i] - response[i]);
    }

    return 2.0 * sum / (n as f64) ;
}