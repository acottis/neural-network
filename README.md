# Neural Network

## Examples

```rust
let mut nn = NeuralNetworkBuilder::<2, 1>::new()
    .epochs(1000000)
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
```

## TODO
Implement callbacks from inside NerualNetwork
