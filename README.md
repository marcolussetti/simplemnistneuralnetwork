# Simple MNIST Neural Network

This library provides a simple neural network implementation using the MNIST handwritten digits dataset and the `simpleneuralnets` library.

It currently does not install on Microsoft Windows due to some compilations issues surrounding the library used for random number generations & wincrypt.h.

## Installation

### Binary (no requirements)
Download the binary from the [releases tab](https://github.com/marcolussetti/simplemnistneuralnetwork/releases) or directly from the instructions below (might be out of date).

```sh
wget https://github.com/marcolussetti/simplemnistneuralnetwork/releases/download/v0.1.2/simplemnistneuralnet
chmod +x simplemnistneuralnet
./simplemnistneuralnet --help
```

### Via Nimble (requires Nim to be installed; installs to Path)
Install Nim from https://github.com/dom96/choosenim#installation

Install the libraries:
```sh
nimble install https://github.com/marcolussetti/mnist_tools
nimble install https://github.com/marcolussetti/simpleneuralnets
nimble install https://github.com/marcolussetti/simplemnistneuralnetwork
```

Run the network:
```sh
simplemnistneuralnet --help
```


## Command Line Interface

Assistance on options is provided by the `--help` parameter

```
$ ./simplemnistneuralnet --help
Usage:
  main [optional-params]
  Options(opt-arg sep :|=|spc):
  -h, --help                               write this help to stdout
  -w, --web_server       bool      false   Enable the web server component, see also port.
  -p=, --port=           int       8080    Set the port to run the web server on. If port 80 is used, remember to use
                                           sudo.
  -t=, --threshold=      float     0.0     The minimum output required to consider the number valid and not an
                                           'Unknown' or exceptional case.
  -r=, --random_batch=   int       0       Randomly searches for ideal hyperparameters, ignores all further
                                           parameters. 0 disables it, otherwise set to the number of searches to
                                           perform.
  -l=, --learning_rate=  float     0.5     Learning rate or alpha for the backpropagation.
  -e=, --epochs=         int       20      The number of rates to train the network for.
  -a=, --activation=     string    "tanh"  The activation function, either 'tanh' for hyperbolic tangent or 'sigmoid'
                                           for logistic sigmoid.
  --hidden_layers=       ,SV[int]  10      The composition of the hidden layers. For instance for 3 hidden layer with
                                           20, 10, and 5 neurons in each, write '20,10,5'
```

## Examples

### Generate a simple neural network
```sh
./simplemnistneuralnet --learning-rate 0.5 --epochs 10 --hidden-layers=20
```

### Serve the simple neural network as a web interface
```sh
./simplemnistneuralnet -w --port 8080 --learning-rate 0.5 --epochs 10 --hidden-layers=20
```

### Generate a pretty good neural network
```sh
./simplemnistneuralnet --learning-rate 0.5 --epochs 53 --hidden-layers=70,85 --activation tanh
```

### Generate hyperparameters via random search
```sh
./simplemnistneuralnet --random-batch 5
```

### Classify uncertain results (<=0.5) as unknown/failed matches
```sh
./simplemnistneuralnet --learning-rate 0.5 --epochs 10 --hidden-layers=20 --threshold 0.5
```

### Full example
```sh
./simplemnistneuralnet -w --port 8080 --learning-rate 0.5 --epochs 55 --hidden-layers=70,85 --activation tanh --threshold 0.5
```

Remember that if running on port 80, you might need to prefix this with sudo.

## Other resources
This project was used as the basis for [a small presentation](https://docs.google.com/presentation/d/1kodOk7US9mpDTi0C_ZwaWYhkhHkKlndwLZTWr7INLVI) discussing the impact of hyperparameters and methods of choosing hyperparameters.
