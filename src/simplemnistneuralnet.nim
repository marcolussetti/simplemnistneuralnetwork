import asynchttpserver, asyncdispatch, sequtils, strutils, sugar, random
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
    randomize()
    let randomNumber = rand(globalTestData.len)

    return $randomNumber

# return randomPage()

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
            intSegment(proc(n: int): auto = ok($n))
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
    random_batch = 0, web_server = false, port = 8080,
    learning_rate = 0.5, epochs = 20, activation = "tanh", threshold = 0.0,
    hidden_layers = @[10],
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
    import cligen; dispatch(main)
