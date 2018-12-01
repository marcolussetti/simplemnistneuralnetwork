import asynchttpserver, asyncdispatch, sequtils, strformat, strutils, sugar
import random
import rosencrantz
import mnist_tools, simpleneuralnets
import ./neuralnet/neuralnet, ./html_templates

var globalNetwork {.threadvar.}: Network
var globalNetworkParams {.threadvar.}: NetworkParameters

var globalTestLabels {.threadvar.}: seq[int]
var globalTestData {.threadvar.}: seq[seq[float]]

proc serverIndexPage(): string =
    return indexPage(globalNetworkParams)

proc serveAllPage(): string =
    let (accuracy, confMatrix) = testMnistNetwork(
            globalNetwork, globalTestData, globalTestLabels,
            globalNetworkParams.threshold
        )
    return allPage(accuracy, confMatrix, globalNetworkParams.threshold)

proc serveRandomPage(): string =
    return randomPage(globalTestLabels.len)

proc serveImagePage(imageNumber: int): string =
    if imageNumber < 0 or imageNumber > globalTestLabels.high:
        return "Image number out of bounds"
    let imageContent = globalTestData[imageNumber]
    let imageLabel = globalTestLabels[imageNumber]
    let denormalizedContent = lc[int(px * 255.0) | (px <- imageContent), int]
    let imageAscii = mnistCoarseAsciiImage(denormalizedContent)

    let predictionVector = classify(globalNetwork, imageContent)
    let prediction = singleOutcome(
        predictionVector, globalNetworkParams.threshold
    )

    let stringPredictionVector = lc[fmt"{x:1.10f}" | (x <- predictionVector),
            string]

    return imagePage(
        imageNumber, imageAscii, imageLabel, prediction,
        stringPredictionVector
    )

proc serveMnistWeb(port: int) =

    let mnistWebServerHandler = get[
        path("/")[
            contentType("text/html")[ok(serverIndexPage())]
        ] ~
        path("/all")[
            contentType("text/html")[ok(serveAllPage())]
        ] ~
        path("/random")[
            contentType("text/html")[ok(serveRandomPage())]
        ] ~
        pathChunk("/image")[
            intSegment(
                proc(n: int): auto = return contentType("text/html")[ok(
                        serveImagePage(n))]
        )
        ]
    ]

    let server = newAsyncHttpServer()
    waitFor server.serve(Port(port), mnistWebServerHandler)


proc setupRandomizedHyperparameterNet*(
    batchSize: int, trainingLabels: seq[int], trainingData: seq[seq[float]],
     testLabels: seq[int], testData: seq[seq[float]], threshold: float
): tuple[a: Network, b: NetworkParameters] =
    var networks = newSeq[tuple[
        network: Network, predictedAccuracy: float, accuracy: float,
        index: int, epochs: int, activation: string, learningRate: float
    ]]()

    for i in 0..<batchSize:
        echo("Random hyperparameter combination #", i)
        let hyperparameters = randomHyperparameters()

        # Train network
        var (network, predictedAccuracy) = simpleMnistNeuralNet(
            hyperparameters.hiddenLayers, hyperparameters.learningRate,
            hyperparameters.epochs, trainingData, trainingLabels,
            hyperparameters.activation
        )

        # Test network
        let (accuracy, confMatrix) = testMnistNetwork(
            network, testData, testLabels
        )

        networks.add((
            network: network, predictedAccuracy: predictedAccuracy,
            accuracy: accuracy, index: i,
            epochs: hyperparameters.epochs,
            activation: hyperparameters.activation,
            learningRate: hyperparameters.learningRate
        ))

        echo("Accuracy on test data: ", accuracy)
        echo("---")

    # Best Performing Network
    var bestPerformingIndex = 0
    for i in 1..<networks.len:
        if networks[i].predictedAccuracy > networks[
                bestPerformingIndex].predictedAccuracy:
            bestPerformingIndex = i
    echo(
        "Best performing network was #", bestPerformingIndex,
        ". Predicted accuracy: ",
        networks[bestPerformingIndex].predictedAccuracy
    )
    echo("Network shape: ", networks[bestPerformingIndex].network)
    echo(
        "Epochs: ", networks[bestPerformingIndex].epochs,
        "; Learning Rate: ", networks[bestPerformingIndex].learningRate,
        "; Activation Function: ",
        networks[bestPerformingIndex].activation
    )
    echo("Accuracy on testing data: ",
        networks[bestPerformingIndex].accuracy)

    var dims = newSeq[int]()
    for layer in networks[bestPerformingIndex].network:
        dims.add(layer[0].weights.len)
    dims.add(
        networks[bestPerformingIndex].network[
            networks[bestPerformingIndex].network.high
        ].len
    )

    return (
        a: networks[bestPerformingIndex].network,
        b: (
            inputLayer: dims[0],
            outputLayer: dims[dims.high],
            hiddenLayers: dims[1..<dims.high],
            learningRate: networks[bestPerformingIndex].learningRate,
            epochs: networks[bestPerformingIndex].epochs,
            activation: networks[bestPerformingIndex].activation,
            threshold: threshold
        )
    )

proc main(
    web_server = false, port = 8080, threshold = 0.0, random_batch = 0,
    learning_rate = 0.5, epochs = 20, activation = "tanh",
    hidden_layers = @[10]
): int =
    randomize()
    let (trainingLabels, trainingData) = normalizeMnistData(mnistTrainingData())
    let (testLabels, testData) = normalizeMnistData(mnistTestData())

    var network: Network
    var networkParameters: NetworkParameters

    if randomBatch == 0:
        var predictedAccuracy: float
        (network, predictedAccuracy) = simpleMnistNeuralNet(
            hiddenLayers, learningRate, epochs,
            trainingData, trainingLabels, activation
        )
        let (accuracy, confMatrix) = testMnistNetwork(
            network, testData, testLabels
        )
        echo("Accuracy on test data: ", accuracy)

        var dims = newSeq[int]()
        for layer in network:
            dims.add(layer[0].weights.len)
        dims.add(network[network.high].len)

        networkParameters = (
            inputLayer: dims[0],
            outputLayer: dims[dims.high],
            hiddenLayers: dims[1..<dims.high],
            learningRate: learningRate,
            epochs: epochs,
            activation: activation,
            threshold: threshold
        )
    else:
        (network, networkParameters) = setupRandomizedHyperparameterNet(
            randomBatch, trainingLabels, trainingData, testLabels,
            testData, threshold
        )

    # Spin up web server if appropriate
    if webServer:
        # Make the network a global variable for async use
        globalNetwork = network
        globalNetworkParams = networkParameters

        # Also export test data to speed up things
        globalTestLabels = testLabels
        globalTestData = testData

        serveMnistWeb(port)

when isMainModule:
    import cligen; dispatch(main, help = {
        "web_server": "Enable the web server component, see also port.",
        "port": "Set the port to run the web server on. If port 80 is used, remember to use sudo.",
        "threshold": "The minimum output required to consider the number valid and not an 'Unknown' or exceptional case.",
        "random_batch": "Randomly searches for ideal hyperparameters, ignores all further parameters. 0 disables it, otherwise set to the number of searches to perform.",
        "learning_rate": "Learning rate or alpha for the backpropagation.",
        "epochs": "The number of rates to train the network for.",
        "activation": "The activation function, either 'tanh' for hyperbolic tangent or 'sigmoid' for logistic sigmoid.",
        "hidden_layers": "The composition of the hidden layers. For instance for 3 hidden layer with 20, 10, and 5 neurons in each, write '20,10,5'"
    })
