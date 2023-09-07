# Neural Network

# Examples

```rust
use rand::{SeedableRng, rngs::StdRng};

let mut nn = nn::NeuralNetwork::new(1, 0.5, StdRng::from_entropy());

nn.layer(5, nn::SIGMOID).layer(1, nn::SIGMOID);

let inputs = vec![vec![1.0], vec![0.0]];
let targets = vec![vec![1.0], vec![0.0]];
nn.train(inputs, targets, 5);

let should_be_0 = nn.feed_forward(vec![0.0])[0];
```


