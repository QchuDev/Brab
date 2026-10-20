use nalgebra::DVector;

pub trait DataProvider {
    fn get_data(&self) -> Vec<TrainData>;
}

#[derive(Clone, Debug)]
pub struct TrainData {
    pub input: DVector<f64>,
    pub expected: DVector<f64>,
}
