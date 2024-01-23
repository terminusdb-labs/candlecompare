mod vecmath;
use std::time::SystemTime;

use candle_core::{Device, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use vecmath::Embedding;

use crate::vecmath::{normalized_cosine_distance, random_normalized_embedding};

fn random_data<R: Rng>(count: usize, rng: &mut R) -> Vec<Embedding> {
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(random_normalized_embedding(rng));
    }

    result
}

fn cpu_compare(v1: &Embedding, v2: &Embedding) -> f32 {
    normalized_cosine_distance(v1, v2)
}

fn tensor_compare(device: &Device, v1: &Embedding, v2: &Embedding) -> f32 {
    let tensor1 = Tensor::new(v1, device).unwrap().reshape((1, 1536)).unwrap();
    let tensor2 = Tensor::new(v2, device).unwrap().reshape((1536, 1)).unwrap();
    let result = tensor1.matmul(&tensor2).unwrap();
    let result = result
        .broadcast_sub(&Tensor::new(1.0f32, device).unwrap())
        .unwrap();
    let result = result
        .broadcast_div(&Tensor::new(-2.0f32, device).unwrap())
        .unwrap();
    let scalar = result.reshape(()).unwrap();
    scalar.to_scalar().unwrap()
}

fn main() {
    let device = Device::Cpu;
    let mut rng = StdRng::seed_from_u64(42);
    let data = random_data(10000, &mut rng);

    let query = random_normalized_embedding(&mut rng);

    let now = SystemTime::now();
    for vec in data.iter() {
        cpu_compare(&query, vec);
    }
    let cpu_duration = now.elapsed().unwrap().as_millis();
    eprintln!("cpu duration: {cpu_duration}");

    let now = SystemTime::now();
    for vec in data.iter() {
        tensor_compare(&device, &query, vec);
    }
    let tensor_duration = now.elapsed().unwrap().as_millis();
    eprintln!("tensor duration: {tensor_duration}");
}
