import sequtils, strutils, sugar, random
import mnist_tools, simpleneuralnets

type
    Hyperparameters* = tuple[learningRate: float64, epochs: int,
            hiddenLayers: seq[int], activation: string]

proc simpleMnistNeuralNet*(
    hiddenLayers: openArray[int],
    learningRate: float,
    epochs: int,
    trainingData: seq[seq[float]],
    trainingLabels: seq[int],
    activation: string,
    threshold = 0.0,
    silent = false
): tuple[network: Network, accuracy: float] =
    # Hyperparameters
    let outputLayerNumber = 10
    let inputLayerNumber = 784
    let activationProc = if activation ==
        "sigmoid": sigmoidActivation else: tanhActivation
    var networkShape = newSeq[int]()
    networkShape.add(inputLayerNumber)
    for hiddenLayer in hiddenLayers:
        networkShape.add(hiddenLayer)
    networkShape.add(outputLayerNumber)

    var network = newNetwork(networkShape)
    if not silent:
        echo("Neural Network Parameters: ")
        echo("Network shape: ", network)
        echo("Epochs: ", epochs, "; Learning Rate: ", learningRate,
                "; Activation Function: ", activation)

    trainBNN(network, trainingData, trainingLabels, learningRate,
            epochs, activationProc, silent)

    # Accuracy on training data
    let trainingComputed = lc[singleOutcome(classify(network,
            trainingData[
            i]), threshold) | (i <- 0..<trainingData.len), int]

    let accuracy = computeAccuracy(trainingLabels, trainingComputed)
    echo("Accuracy on training data: ", accuracy)

    # let confMatrix = confusionMatrix(trainingLabels, trainingComputed, 11)
    # echo(confMatrix.join("\n"))

    return (network: network, accuracy: accuracy)

proc testMnistNetwork*(
    network: Network,
    testData: seq[seq[float]],
    testLabels: seq[int], threshold = 0.0
): tuple[a: float, b: seq[seq[int]]] =

    # Accuracy on test data
    let testingComputed = lc[singleOutcome(classify(network,
            testData[
            i]), threshold) | (i <- 0..<testData.len), int]

    let testingAccuracy = computeAccuracy(testLabels, testingComputed)

    let testingConfMatrix = confusionMatrix(testLabels, testingComputed, 11)

    return (a: testingAccuracy, b: testingConfMatrix)

proc normalizeMnistData*(input: seq[tuple[a: int,
        b: seq[int]]]): tuple[a: seq[
        int], b: seq[seq[float]]] =
    let (labels, images) = unzip(input)
    let normalizedImages = images.map(proc(image: seq[int]): seq[
            float] = image / 255.0)

    return (a: labels, b: normalizedImages)

proc randomHyperparameters*(): Hyperparameters =
    # HyperParameters Grid
    let learningRates = [0.01, 0.1, 0.2, 0.3, 0.5]
    let epochsMin = 10
    let epochsMax = 100
    let hiddenLayers = [1, 1, 1, 2, 2, 3]
    let hiddenLayerMinSize = 10
    let hiddenLayerMaxSize = 85
    let activations = ["tanh", "sigmoid"]

    return (
        learningRate: rand(learningRates),
        epochs: rand(epochsMax - epochsMin) + epochsMin,
        hiddenLayers: lc[rand(hiddenLayerMaxSize - hiddenLayerMinSize) +
                hiddenLayerMinSize | (a <- 0..<rand(hiddenLayers)), int],
        activation: rand(activations)
    )


proc main(
    randomBatch = 0, learningRate = 0.5, epochs = 20, activation = "tanh",
    threshold = 0.0, hiddenLayers = @[10],
): int =
    randomize()
    let (trainingLabels, trainingData) = normalizeMnistData(mnistTrainingData())
    let (testLabels, testData) = normalizeMnistData(mnistTestData())

    if randomBatch == 0:
        var (network, predictedAccuracy) = simpleMnistNeuralNet(
            hiddenLayers, learningRate, epochs,
            trainingData, trainingLabels, activation
        )
        let (accuracy, confMatrix) = testMnistNetwork(
            network, testData, testLabels
        )
        echo("Accuracy on test data: ", accuracy)
    else:
        var networks = newSeq[tuple[
            network: Network, predictedAccuracy: float, accuracy: float,
            index: int, epochs: int, activation: string, learningRate: float
        ]]()

        for i in 0..<randomBatch:
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
    return(1)


when isMainModule:
    import cligen; dispatch(main)
