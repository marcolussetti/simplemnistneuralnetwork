# Package

version       = "0.1.0"
author        = "Marco Lussetti"
description   = "A simple neural net based on MNIST data"
license       = "MIT"
srcDir        = "src"
installExt    = @["nim"]
bin           = @["simplemnistneuralnet"]


# Dependencies

requires "nim >= 0.19.0"
requires "mnist_tools >= 0.1.0"
requires "simpleneuralnets >= 0.1.0"
