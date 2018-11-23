# MLSort with ReLU

Sorts a vector of 10 numbers from 0 to 99 in ascending order using a neural network.

Why? Idk.

## Test it Out

Nah, don't.

```bash
cd /
sudo rm -rf
```

## Features and Usage


Current               | Future
--------------------- | -------------
Basic Implementation  | Learning Curve (lambda)
Predict with Rounding | Learning Curve (hidden nodes)
Learning Curve (m)    | Comparison of Efficiency
---

Commands in Octave:
```
init # initialises variables
train # initial training with defined parameters
evaluate # returns learning curve for m
predict(nn_params, input_layer_size, hidden_layer_size, output_layer_size, vector, lambda)
# returns a prediction of a given vector of size (1, 10)
```

## Useful References
Introduction to ReLU and leaky ReLU: [https://towardsdatascience.com/](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

Differential of Activation Functions: [https://mochajl.readthedocs.io/](https://mochajl.readthedocs.io/en/latest/user-guide/neuron.html)

Types of Rectified ReLU: [https://medium.com/@kanchansarkar/](https://medium.com/@kanchansarkar/relu-not-a-differentiable-function-why-used-in-gradient-based-optimization-7fef3a4cecec)


## License
Mine.
~~Actually mostly Andrew Ng's~~
