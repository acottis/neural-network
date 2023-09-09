use std::f64::consts::E;

use rand::{rngs::StdRng, Rng};

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Debug)]
struct Layer {
    /// A collection of Weights and Biases, each neuron has the same number of weights as a layer
    /// has inputs
    neurons: Vec<Neuron>,

    /// The inputs to the current layer before any operations
    inputs: Vec<f64>,

    /// The outputs of a layer before activation function is applied
    outputs: Vec<f64>,

    /// The activation function that is applied on this layer
    activation: Activation,
}

impl Layer {
    fn new(rng: &mut StdRng, inputs: usize, neurons: usize, activation: Activation) -> Self {
        let mut neurons = Vec::with_capacity(neurons);
        let neurons_count = neurons.capacity();

        for _ in 0..neurons_count {
            let mut weights = Vec::with_capacity(inputs);

            for _ in 0..inputs {
                weights.push(rng.gen());
            }
            neurons.push(Neuron { weights, bias: 0.0 });
        }

        Self {
            neurons,
            inputs: Vec::with_capacity(inputs),
            outputs: vec![0.0; neurons_count],
            activation,
        }
    }

    /// This is the hot loop
    fn forward_pass(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        // TODO: This needs to go
        let mut layer_outputs = Vec::with_capacity(self.neurons.len());

        // Save our inputs for backward propagation
        self.inputs.clear();
        self.inputs.extend(inputs);

        for (neuron, output) in self.neurons.iter().zip(self.outputs.iter_mut()) {
            // Since we are adding below we need to reset output to 0.0 as we are mutating
            // an exisiting value
            *output = 0.0;

            // For every weight and every input dot product
            for (weight, input) in neuron.weights.iter().zip(inputs.iter()) {
                *output += weight * input;
            }

            // Save our output before activation for use in cost calculation
            *output += neuron.bias;

            // Activate our neuron and add to our return
            layer_outputs.push((self.activation.function)(*output));
        }
        layer_outputs
    }

    /// Calculate our errors and update weights/biases
    fn backward_pass(&mut self, learning_rate: f64, mut errors: Vec<f64>) -> Vec<f64> {
        let deltas: Vec<f64> = self
            .outputs
            .iter()
            .zip(errors.iter())
            .map(|(output, error)| error * (self.activation.derivative)(*output))
            .collect();

        // Errors no longer needed so we re-init
        errors.clear();

        for (neuron, delta) in self.neurons.iter_mut().zip(deltas.iter()) {
            // Update bias
            neuron.bias -= learning_rate * delta;
            for (weight, input) in neuron.weights.iter_mut().zip(self.inputs.iter()) {
                // Calculate error for next layer
                errors.push(*weight * delta);

                // Update weights
                *weight -= learning_rate * delta * input;
            }
        }

        errors
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    inputs_len: usize,
    rng: StdRng,
}

impl NeuralNetwork {
    /// # Examples
    /// ```
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let nn = nn::NeuralNetwork::new(2, 0.05, StdRng::seed_from_u64(1337));
    /// ```
    ///
    /// ```
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let nn = nn::NeuralNetwork::new(4, 0.001, StdRng::from_entropy());
    /// ```
    pub fn new(inputs_len: usize, learning_rate: f64, rng: StdRng) -> Self {
        Self {
            layers: Vec::new(),
            learning_rate,
            rng,
            inputs_len,
        }
    }

    pub fn layer(&mut self, neurons: usize, activation: Activation) -> &mut Self {
        let inputs_len = match self.layers.last() {
            Some(layer) => layer.neurons.len(),
            None => self.inputs_len,
        };

        self.layers
            .push(Layer::new(&mut self.rng, inputs_len, neurons, activation));
        self
    }

    /// # Examples
    ///
    /// ```
    /// use rand::{SeedableRng, rngs::StdRng};
    ///
    /// let mut nn = nn::NeuralNetwork::new(1, 0.5, StdRng::from_entropy());
    ///
    /// nn.layer(5, nn::SIGMOID).layer(1, nn::SIGMOID);
    ///
    /// let inputs = vec![vec![1.0], vec![0.0]];
    /// let targets = vec![vec![1.0], vec![0.0]];
    /// nn.train(inputs, targets, 5);
    /// ```
    pub fn train(
        &mut self,
        training_inputs: Vec<Vec<f64>>,
        training_targets: Vec<Vec<f64>>,
        epochs: usize,
    ) {
        if self.layers.len() == 0 {
            panic!("No layers added")
        }
        for epoch in 0..epochs {
            for (inputs, targets) in training_inputs.iter().zip(training_targets.iter()) {
                // Get the prediction for the training run
                let outputs = self.feed_forward(inputs.clone());

                //                if epoch % 1000 == 0 {
                //                    println!(
                //                        "Epoch: {} inputs: {:?}, predictions: {:?}",
                //                        epoch, &inputs, &outputs
                //                    );
                //                }

                // Update our weights based on how far prediction is from expected
                self.back_propagate(&outputs, &targets);
            }
        }
    }

    fn back_propagate(&mut self, outputs: &Vec<f64>, targets: &Vec<f64>) {
        let mut errors: Vec<f64> = outputs
            .iter()
            .zip(targets.iter())
            .map(|(output, target)| output - target)
            .collect();

        for layer in self.layers.iter_mut().rev() {
            errors = layer.backward_pass(self.learning_rate, errors);
        }
    }

    /// Returns our predicition for a layer
    pub fn feed_forward(&mut self, mut inputs: Vec<f64>) -> Vec<f64> {
        assert!(inputs.len() == self.inputs_len);
        for layer in self.layers.iter_mut() {
            inputs = layer.forward_pass(&inputs);
        }
        inputs
    }
}

#[derive(Debug)]
pub struct Activation {
    function: fn(f64) -> f64,
    derivative: fn(f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| (SIGMOID.function)(x) * (1.0 - (SIGMOID.function)(x)),
};

pub const IDENTITY: Activation = Activation {
    function: |x| x * 1.0,
    derivative: |_| 1.0,
};

pub const RELU: Activation = Activation {
    function: |x| x.max(0.0),
    derivative: |x| if x > 0.0 { 1.0 } else { 0.0 },
};

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn one_sized_nn() {
        let mut nn = NeuralNetwork::new(1, 0.5, StdRng::from_entropy());
        nn.layer(5, SIGMOID).layer(1, SIGMOID);

        let inputs = vec![vec![1.0], vec![0.0]];
        let targets = vec![vec![1.0], vec![0.0]];

        nn.train(inputs, targets, 25000);

        let should_be_0 = nn.feed_forward(vec![0.0])[0];
        assert!(should_be_0.round() == 0.0);

        let should_be_1 = nn.feed_forward(vec![1.0])[0];
        assert!(should_be_1.round() == 1.0);
    }
}
