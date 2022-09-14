#![allow(non_snake_case)]

use ndarray::{arr2, Array, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

struct LayerDense {
    weights: Array2<f32>,
    biases: Array2<f32>,
}

impl LayerDense {
    fn new(num_inputs: usize, num_neurons: usize) -> Self {
        let weights = Array::random((num_inputs, num_neurons), Uniform::new(0., 1.)) * 0.2;
        let biases = Array::zeros((1, num_neurons));
        Self { weights, biases }
    }

    fn forward(&self, inputs: &Array2<f32>) -> Array2<f32> {
        inputs.dot(&self.weights) + &self.biases
    }
}

fn activation_relu(inputs: &Array2<f32>) -> Array2<f32> {
    inputs.map(|n| n.max(0.))
}

// from https://github.com/Sentdex/NNfSiX/blob/master/Python/p006-Softmax-Activation.py
fn softmax(input: Array2<f64>) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros(input.raw_dim());
    for (in_row, mut out_row) in input.axis_iter(Axis(0)).zip(output.axis_iter_mut(Axis(0))) {
        let mut max = 0.0;
        for col in in_row.iter() {
            if col > &max {
                max = *col;
            }
        }
        let exp = in_row.map(|x| (x - max).exp());
        let sum = exp.sum();
        out_row.assign(&(exp / sum));
    }
    output
}

fn main() {
    let X = arr2(&[
        [1., 2., 3., 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]);

    let layer1 = LayerDense::new(4, 5);
    let layer2 = LayerDense::new(5, 2);

    let output = layer1.forward(&X);
    let output = layer2.forward(&output);
    dbg!(&output);
}
