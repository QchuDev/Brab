use core::f64;

use nalgebra::DVector;
use brab::{
    activation::Sigmoid, 
    cost::Mse, 
    network::NeuralNetwork
};

include!("mnist_data.rs");

fn main() {
    println!("Start..\n");
    let mut net = NeuralNetwork::new(&Vec::from([784, 10, 10, 10]), Sigmoid::new(), Mse::new());

    let mnist_prov = MnistProvider::new();

    let _ = net.save("model_0");
    NeuralNetwork::new_load("model_0");
    
    // run it for some epochs and it will learn
    for _ in 0..1 {
        println!("{}", net.train_multiple(0.06f64, mnist_prov.get_data()).unwrap());
    }

    let _ = net.save("model_1");

}



// use safetensors::{Dtype, SafeTensors, serialize_to_file, tensor::TensorView};
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // 4. Leer el archivo.
//     let archivo = std::fs::read("datos.safetensors")?;
//     let tensors = SafeTensors::deserialize(&archivo)?;

//     // 5. Recuperar el tensor por su nombre.
//     let tensor_leido = tensors.tensor("mi_informacion")?;

//     // 6. Convertir los bytes nuevamente a f32.
//     let recuperados: Vec<f32> = tensor_leido
//         .data()
//         .chunks_exact(4)
//         .map(|chunk| {
//             f32::from_le_bytes([
//                 chunk[0],
//                 chunk[1],
//                 chunk[2],
//                 chunk[3],
//             ])
//         })
//         .collect();

//     println!("Valores recuperados: {:?}", recuperados);

//     Ok(())
// }
