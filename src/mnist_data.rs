use brab::data::{DataProvider, TrainData};
use mnist_reader::{MnistReader};

pub struct MnistProvider { reader:MnistReader }
impl MnistProvider {
    pub fn new() -> Self {
        let mut mnist = MnistReader::new("mnist-data");
        mnist.load().unwrap();
        
        Self {
            reader: mnist
        }
    }
    pub fn get_raw_data(&self) -> &Vec<Vec<f32>> {
        return &self.reader.train_data;
    }
    pub fn get_labels(&self) -> &Vec<u8> {
        return &self.reader.train_labels;
    }    
}
impl DataProvider for MnistProvider{
    fn get_data(&self) -> Vec<TrainData> {
        let labeled_vectors = self.get_raw_data()
            .into_iter()
            .zip(self.get_labels())
            .map(|(data, label)| TrainData { expected: DVector::from_fn(10, |i,j| -> f64 {if *label == (i as u8){1.0} else {0.0}}), input: DVector::from(data.clone()).cast::<f64>() })
            .collect();

        return labeled_vectors;
    }
}
