import templates

type
    NetworkParameters* = tuple[
        inputLayer: int,
        outputLayer: int,
        hiddenLayers: seq[int],
        learningRate: float,
        epochs: int,
        activation: string,
        threshold: float
    ]

proc indexPage*(networkParams: NetworkParameters): string = tmpli html"""
<html>
    <head>
        <title>COMP3710: Handwritten digits classification</title>
    </head>
    <body>
        <h1>COMP3710: Handwritten digits classification</h1>
        <div class="header">
            <p>
                This project was developed as a final project for the
                Artificial Intelligence (COMP3710) course at Thompson
                Rivers University under the supervision of Dr. Lee.
            </p>
            <p>
                It uses a simple backpropagation neuralnetwork with delta rule
                to classify handwritten digits from the
                <a href="http://yann.lecun.com/exdb/mnist/">MNIST dataset</a>.
                The entire project was written in the programming language
                <a href="http://nim-lang.org">Nim</a>.
            </p>
            <p>
                This project was created by
                <a href="https://marcolussetti.com">Marco Lussetti</a> as an
                open source project.
            </p>
            <div>
                You can find the project on GitHub as a set of libraries:
                <ul>
                    <li>
                        <a href="https://github.com/marcolussetti/mnist_tools">
                        mnist_tools</a>: this library allows one to download and
                        manipulate the aforementioned MNIST dataset.
                    </li>
                    <li>
                        <a href="https://github.com/marcolussetti/simpleneuralnets">
                        simpleneuralnets</a>: this library offers a very simple
                        interface to create basic neural networks.
                    </li>
                    <li>
                        <a href="https://github.com/marcolussetti/simplemnistneuralnetwork">
                        simplemnistneuralnetwork</a>: this is the package you
                        see currently! It trains a neural network from the
                        simpleneuralnets library with the MNIST dataset, and
                        serves this website over a basic webserver for you to
                        test. Alternatively, it may be run in command-line
                        mode.
                    </li>
                </ul>
            </div>
        </div>
        <hr />
        <div class="content">
            <div>
                This are the hyperparameters this neural network was trained with:
                <ul>
                    <li>Learning Rate: $(networkParams.learningRate)</li>
                    <li>Epochs: $(networkParams.epochs)</li>
                    $if networkParams.activation == "tanh" {
                        <li>Activation function: Hyperbolic tangent ("tanh")</li>
                    }
                    $elif networkParams.activation == "sigmoid" {
                        <li>Activation function: Logistic Sigmoid ("sigmoid")</li>
                    }
                    <li>Network Shape:
                        <ul>
                            <li>Input Layer: $(networkParams.inputLayer)</li>
                            $for i in 0..<networkParams.hiddenLayers.len {
                                <li>Hidden Layer #$(i + 1): $(networkParams.hiddenLayers[i])</li>
                            }
                            <li>Output Layer: $(networkParams.outputLayer)</li>
                        </ul>
                    </li>
                    <li>The threshold for accepting a result (rather than returning unknown) is: $(networkParams.threshold)</li>
                </ul>
            </div>
        </div>
        <hr />
        <div class="test">
            <p>You can test this network yourself, using MNIST's test dataset!</p>
            <div class="test-entire">
                <p>
                    Test network on the entire MNIST test dataset
                    (10,000 images). This will return an accuracy score and a
                    confusion matrix.
                </p>
                <p><b><a href="/all">Test on entire test dataset!</a></b></p>
            </div>
            <br />
            <div class="test-single">
                <p>
                    Test network on a single image. This will show you an
                    image from the MNIST dataset as a basic ASCII printout,
                    and ask the network to classify it and show the
                    prediction.
                </p>
                <p><b><a href="/random">Test on a random image!</a></b></p>
                <p><b>
                    Select specific image by number [0-9999]:
                    <input id="image" value="4"></input>
                    <button id="image-btn">Test Image!</button>
                </b></p>
            </div>
        </div>
        <script>
            document.querySelector("#image-btn").addEventListener("click", function() {
                var targetImage = document.querySelector("#image").value;
                window.location.href = "/image/" + targetImage;
            });
        </script>
    </body>
</html>
"""

proc allPage*(
    accuracy: float, confusionMatrix: seq[seq[int]], threshold: float
): string = tmpli html"""
    <html>
        <head>
            <title>COMP3710 - Handwritten digits: Test on all test data</title>
            <style>
                table, th, td { border: 1px solid black; }
            </style>
        </head>
        <body>
            <h1>COMP3710 - Handwritten digits: Test on all test data</h1>
            <div class="header">
                <p>
                    We have just tested the 10,000 images in the test dataset
                    against our neural network (that's why loading the page
                    took a while!).
                </p>
                <p>
                    We used a threshold of $threshold. This means that if none
                    of the confidence values was above $threshold, we
                    treated the image as being not a digit.
                </p>
                <p>Here are the results:</p>
            </div>
            <div class="accuracy">
                <h2>Accuracy</h2>
                <b>Accuracy:</b> $accuracy
            </div>
            <div class="confusion-matrix">
                <h2>Confusion Matrix</h2>
                <p>
                    This visualizes the hit and misses of the network's
                    classification.
                </p>
                <div class="matrix">
                    <table>
                    <tr>
                        <th rowspan="$(confusionMatrix.len + 1 + 2)">
                            Actual
                        </th>
                    </tr>
                    <tr>
                        <th colspan="$(confusionMatrix[0].len + 1 + 1)">
                            Predicted
                        </th>
                    </tr>
                    <tr>
                        <th></th>
                        $for i in 0..<confusionMatrix[0].len {
                            <th>
                                $if i > 0 {
                                    $(i - 1)
                                }
                                $else {
                                    Unknown
                                }
                            </th>
                        }
                    </tr>
                    $for i in 0..<confusionMatrix.len {
                        <tr>
                            <th>
                                $if i > 0 {
                                    $(i - 1)
                                }
                                $else {
                                    Unknown
                                }
                            </th>
                            $for j in 0..<confusionMatrix[0].len {
                                <td>$(confusionMatrix[i][j])</td>
                            }
                        </tr>
                    }
                    </table>
                </div>
            </div>
        </body>
    </html>
"""
