use std::f64::consts::E;

use log::info;
use rand::{rngs::StdRng, Rng, SeedableRng};

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
    fn forward_pass(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // TODO: This needs to go
        // However, rust seems to optimise this out from strace
        let mut activateds = Vec::with_capacity(self.neurons.len());

        // Save our inputs for backward propagation
        self.inputs.clear();
        self.inputs.extend(inputs.iter());

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
            activateds.push((self.activation.function)(*output));
        }
        activateds
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
pub struct NeuralNetwork<const INPUTS_LEN: usize, const OUTPUTS_LEN: usize> {
    layers: Vec<Layer>,
    learning_rate: f64,
    cost_function: CostFunction,
}

impl<const INPUTS_LEN: usize, const OUTPUTS_LEN: usize> NeuralNetwork<INPUTS_LEN, OUTPUTS_LEN> {
    /// # Examples
    ///
    /// ```
    /// use rand::{Rng, rngs::StdRng, SeedableRng};
    /// use nn::*;
    ///
    /// let mut nn = NeuralNetworkBuilder::<2, 1>::new()
    ///     .epochs(1000000)
    ///     .rng(StdRng::seed_from_u64(1337))
    ///     .learning_rate(0.00005)
    ///     .output_layer(IDENTITY);
    ///
    /// let mut inputs = Vec::new();
    /// let mut targets = Vec::new();
    /// let mut rng = StdRng::seed_from_u64(1337);
    ///
    /// (0..100).for_each(|_| {
    ///     let rand1 = rng.gen_range(0.0..10.0);
    ///     let rand2 = rng.gen_range(0.0..rand1);
    ///     let input = [rand2, rand1 - rand2];
    ///     targets.push([input[0] + input[1]]);
    ///     inputs.push(input);
    /// });
    ///
    /// nn.train(inputs, targets, 100);
    /// ```
    ///
    /// # panics!
    /// This function will panic if you pass training_inputes and training_targets
    /// of different lengths.
    pub fn train(
        &mut self,
        training_inputs: Vec<[f64; INPUTS_LEN]>,
        training_targets: Vec<[f64; OUTPUTS_LEN]>,
        epochs: usize,
    ) {
        assert!(
            training_inputs.len() == training_targets.len(),
            "training_inputs ({}) must be same length as training_targets ({})",
            training_inputs.len(),
            training_targets.len()
        );

        for epoch in 0..epochs {
            for (inputs, targets) in training_inputs.iter().zip(training_targets.iter()) {
                // Get the prediction for the training run
                let predictions = self.feed_forward(*inputs);

                // Update our weights based on how far prediction is from expected
                let cost = self.back_propagate(predictions, &targets);

                if epoch < 1000 || epoch % 1000 == 0 {
                    info!("EPOCH {epoch:2}: Cost {cost}");
                }
            }
        }
    }

    fn back_propagate(
        &mut self,
        activateds: [f64; OUTPUTS_LEN],
        targets: &[f64; OUTPUTS_LEN],
    ) -> f64 {
        let mut errors: Vec<f64> = activateds
            .iter()
            .zip(targets.iter())
            .map(|(activated, target)| (self.cost_function.derivative)(*activated, *target))
            .collect();

        for layer in self.layers.iter_mut().rev() {
            errors = layer.backward_pass(self.learning_rate, errors);
        }

        let cost: f64 = activateds
            .iter()
            .zip(targets.iter())
            .map(|(activated, target)| (self.cost_function.function)(*activated, *target))
            .reduce(|acc, cost| acc + cost)
            .unwrap();
        let average_cost = cost / activateds.len() as f64;
        average_cost
    }

    /// Returns our predicition for a layer
    pub fn feed_forward(&mut self, inputs: [f64; INPUTS_LEN]) -> [f64; OUTPUTS_LEN] {
        let mut inputs = inputs.to_vec();

        // For every layer, pass the inputs from the previous
        // the size of inputs can change, but rust knows at compile time
        // which size it will be at every iteration and can optimise out
        for layer in self.layers.iter_mut() {
            inputs = layer.forward_pass(inputs);
        }
        // This is the activated neurons
        let mut prediction = [0.0_f64; OUTPUTS_LEN];
        prediction.copy_from_slice(&inputs);

        // Return the answer to the caller
        prediction
    }
}

#[derive(Debug)]
/// # Examples
/// ```
/// use rand::{rngs::StdRng, SeedableRng};
/// use nn::*;
///
/// let mut nn = NeuralNetworkBuilder::<2, 1>::new()
///     .epochs(1000000)
///     .rng(StdRng::seed_from_u64(1337))
///     .learning_rate(0.00005)
///     .layer(5, SIGMOID)
///     .output_layer(IDENTITY);
/// ```
pub struct NeuralNetworkBuilder<const INPUTS_LEN: usize, const OUTPUTS_LEN: usize> {
    learning_rate: f64,
    epochs: usize,
    rng: StdRng,
    cost_function: CostFunction,
    layer_layout: Vec<(usize, Activation)>,
}

impl<const INPUTS_LEN: usize, const OUTPUTS_LEN: usize>
    NeuralNetworkBuilder<INPUTS_LEN, OUTPUTS_LEN>
{
    /// Create a builder, intialised with our Neural Network
    /// defaults and can be configured by calling its member functions
    /// Call [Self::output_layer()] to complete and return a [NeuralNetwork]
    pub fn new() -> Self {
        Self {
            learning_rate: 0.025,
            epochs: 1,
            rng: StdRng::from_entropy(),
            cost_function: MEAN_SQUARED,
            layer_layout: Vec::new(),
        }
    }

    /// The amount our back propagation changes the weights
    /// A high number means we learn faster but can cause issues
    /// with the model overshooting the target weights
    /// default is 0.025
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// The number of times the same training data will be run
    /// against our model, the more we train the more danger of overfitting
    /// we run in to in complex problems. Default is 1
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// The type of [StdRng] to use for seeding the weights
    /// Can be set to a seed with [StdRng::seed_from_u64()] to get deterministic runs
    pub fn rng(mut self, rng: StdRng) -> Self {
        self.rng = rng;
        self
    }

    /// Add a layer to the neural network
    pub fn layer(mut self, neurons: usize, activation: Activation) -> Self {
        self.layer_layout.push((neurons, activation));
        self
    }

    /// Finialise the Neural network with the output layer, this then builds
    /// your [NeuralNetwork] and returns that [NeuralNetwork]
    pub fn output_layer(
        mut self,
        output_activation: Activation,
    ) -> NeuralNetwork<INPUTS_LEN, OUTPUTS_LEN> {
        let mut layers = Vec::with_capacity(self.layer_layout.len() + 1);

        let mut inputs_len = INPUTS_LEN;
        for layer in self.layer_layout {
            // Initialise the weights and biases of one layer
            layers.push(Layer::new(&mut self.rng, inputs_len, layer.0, layer.1));

            // The next layer has the inputs_len of the neurons of the current layer
            inputs_len = layer.0;
        }
        // Create out output layer
        layers.push(Layer::new(
            &mut self.rng,
            inputs_len,
            OUTPUTS_LEN,
            output_activation,
        ));

        NeuralNetwork {
            layers,
            learning_rate: self.learning_rate,
            cost_function: self.cost_function,
        }
    }
}

#[derive(Debug)]
pub struct CostFunction {
    function: fn(activated: f64, target: f64) -> f64,
    derivative: fn(activated: f64, target: f64) -> f64,
}

pub const MEAN_SQUARED: CostFunction = CostFunction {
    function: |activated, target| (activated - target).powi(2),
    derivative: |activated, target| activated - target,
};

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
    use super::*;

    #[test]
    fn builder_nn_size_1_1() {
        let nn = NeuralNetworkBuilder::<1, 1>::new()
            .learning_rate(0.5)
            .output_layer(IDENTITY);

        assert!(nn.layers.len() == 1, "Only one layer created");
        assert!(
            nn.layers[0].neurons.len() == 1,
            "Only one output neuron created"
        );
    }

    #[test]
    fn builder_correct_layer_sizes() {
        let nn = NeuralNetworkBuilder::<6, 25>::new()
            .learning_rate(0.5)
            .layer(10, SIGMOID)
            .layer(5, RELU)
            .output_layer(IDENTITY);

        assert!(nn.layers.len() == 3, "{}", nn.layers.len());
        assert!(nn.layers[0].neurons.len() == 10);
        assert!(nn.layers[0].neurons[0].weights.len() == 6);

        assert!(nn.layers[1].neurons.len() == 5);
        assert!(nn.layers[1].neurons[0].weights.len() == 10);

        assert!(nn.layers[2].neurons.len() == 25);
        assert!(nn.layers[2].neurons[0].weights.len() == 5);

        assert!(
            nn.layers.get(3).is_none(),
            "There should not be a 4th layer"
        );
    }
    #[test]
    fn builder_learning_rate() {
        const LEARNING_RATE: f64 = 46.7;
        let nn = NeuralNetworkBuilder::<1, 1>::new()
            .learning_rate(LEARNING_RATE)
            .output_layer(IDENTITY);

        assert!(
            nn.learning_rate == LEARNING_RATE,
            "Learning rate set correctly"
        );
    }

    #[test]
    fn nn_one_input_one_output() {
        let mut nn = NeuralNetworkBuilder::<1, 1>::new().output_layer(IDENTITY);

        let inputs = vec![[1.0], [0.0]];
        let targets = vec![[1.0], [0.0]];

        nn.train(inputs, targets, 250);

        let should_be_0 = nn.feed_forward([0.0])[0];
        assert!(should_be_0.round() == 0.0, "{}", should_be_0.round());

        let should_be_1 = nn.feed_forward([1.0])[0];
        assert!(should_be_1.round() == 1.0, "{}", should_be_1.round());
    }

    #[test]
    fn nn_sum_2_numbers() {
        let mut nn = NeuralNetworkBuilder::<2, 1>::new()
            .epochs(1000)
            .learning_rate(0.00005)
            .output_layer(IDENTITY);

        let mut rng = StdRng::from_entropy();
        let mut inputs = Vec::with_capacity(10000);
        let mut targets = Vec::with_capacity(10000);

        (0..100).for_each(|_| {
            let rand1 = rng.gen_range(0.0..10.0);
            let rand2 = rng.gen_range(0.0..rand1);
            let input = [rand2, rand1 - rand2];
            targets.push([input[0] + input[1]]);
            inputs.push(input);
        });

        nn.train(inputs, targets, 100);

        let should_be_6 = nn.feed_forward([4.0, 2.0])[0];
        assert!(should_be_6.round() == 6.0, "{}", should_be_6.round());

        let should_be_3 = nn.feed_forward([1.0, 2.0])[0];
        assert!(should_be_3.round() == 3.0, "{}", should_be_3.round());
    }
}
