use std::f64::consts::E;

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
    activation: Activation,
}

impl Layer {
    fn new(rng: &mut StdRng, inputs_len: usize, neurons: usize, activation: Activation) -> Self {
        let mut layer = Vec::with_capacity(neurons);

        for _ in 0..neurons {
            let mut weights = Vec::with_capacity(inputs_len);
            for _ in 0..inputs_len {
                weights.push(rng.gen());
            }

            layer.push(Neuron { weights, bias: 0.0 });
        }

        Self {
            neurons: layer,
            inputs: Vec::with_capacity(inputs_len),
            outputs: vec![0.0; inputs_len],
            activation,
        }
    }

    fn forward_pass(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        // TODO: This needs to go
        let mut activated_outputs = Vec::with_capacity(self.neurons.len());

        // Save our inputs for backward propagation
        self.inputs.clear();
        self.inputs.extend_from_slice(&inputs);

        for (neuron, output) in self.neurons.iter().zip(self.outputs.iter_mut()) {
            *output = 0.0;
            // For every weight and every input dot product
            for (weight, input) in neuron.weights.iter().zip(inputs.iter()) {
                *output += weight * input;
            }
            // Save our output before activation
            *output += neuron.bias;
            activated_outputs.push(self.activation.function(*output));
        }
        activated_outputs
    }

    fn backward_pass(&mut self, learning_rate: f64, mut errors: Vec<f64>) -> Vec<f64> {
        let deltas: Vec<f64> = self
            .outputs
            .iter()
            .zip(errors.iter())
            .map(|(output, error)| error * self.activation.derivative(*output))
            .collect();

        // Errors no longer needed so we re-init
        errors = Vec::with_capacity(self.inputs.len());

        for (neuron, (delta, input)) in self
            .neurons
            .iter_mut()
            .zip(deltas.iter().zip(self.inputs.iter()))
        {
            // Update bias
            neuron.bias -= learning_rate * delta;
            for weight in neuron.weights.iter_mut() {
                // Calculate error for next layer
                errors.push(*weight * delta);

                // Update weights
                *weight -= learning_rate * delta * input;
            }
        }

        errors
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        let learning_rate = 0.5;
        let mut rng = StdRng::seed_from_u64(1337);

        let layers = vec![
            Layer::new(&mut rng, 1, 10, Activation::Sigmoid),
            Layer::new(&mut rng, 10, 10, Activation::Sigmoid),
            Layer::new(&mut rng, 10, 1, Activation::Sigmoid),
        ];

        Self {
            layers,
            learning_rate,
        }
    }

    pub fn train(
        &mut self,
        training_inputs: Vec<Vec<f64>>,
        training_targets: Vec<Vec<f64>>,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            for (inputs, targets) in training_inputs.iter().zip(training_targets.iter()) {
                let outputs = self.feed_forward(inputs.clone());

                if epoch % 100 == 0 {
                    println!(
                        "Epoch: {epoch} inputs: {:?}, predictions: {:?}",
                        &inputs, &outputs
                    );
                }

                self.back_propagate(&outputs, &targets);
            }
        }
    }

    fn back_propagate(&mut self, outputs: &Vec<f64>, targets: &Vec<f64>) {
        let mut errors: Vec<f64> = targets
            .iter()
            .zip(outputs.iter())
            .map(|(target, output)| output - target)
            .collect();

        for layer in self.layers.iter_mut().rev() {
            errors = layer.backward_pass(self.learning_rate, errors);
        }
    }

    /// Returns our predicition for a layer
    pub fn feed_forward(&mut self, mut inputs: Vec<f64>) -> Vec<f64> {
        for layer in self.layers.iter_mut() {
            inputs = layer.forward_pass(&inputs);
        }
        inputs
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum Activation {
    Sigmoid,
    Identity,
}

impl Activation {
    fn function(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
            Self::Identity => x * 1.0,
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => self.function(x) * (1.0 - self.function(x)),
            Self::Identity => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_sized_nn() {
        let mut nn = NeuralNetwork::new();

        let inputs = vec![vec![1.0], vec![0.0]];
        let targets = vec![vec![1.0], vec![0.0]];

        nn.train(inputs, targets, 25000);

        let should_be_0 = nn.feed_forward(vec![0.0])[0];
        assert!(should_be_0.round() == 0.0);

        let should_be_1 = nn.feed_forward(vec![1.0])[0];
        assert!(should_be_1.round() == 1.0);
    }
}
