/// This CostFunction trait expects you to implement: the function itself and its gradient respect to a (activation)
pub trait CostFunction {
    fn function(&self, expected: &DVector<f64>, response: &DVector<f64>) -> f64;
    fn gradient_a(&self, expected: &DVector<f64>, response: &DVector<f64>) -> Option<DVector<f64>>;
    
}

/*  
================        COST FUNCTIONS        ================
*/
struct Mse {}
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