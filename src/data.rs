use nalgebra::DVector;

pub trait DataProvider {
    fn get_data(&self) -> Vec<TrainData>;
}

#[derive(Clone, Debug)]
pub struct TrainData {
    pub input: DVector<f64>,
    pub expected: DVector<f64>,
}

/// Post train data results are returned in training functions to call the state of
/// the current training session, some statistics will be collected on creation.

pub struct TrainResults {
    // accuracy
    // error
    // certainty
    // ...
}
