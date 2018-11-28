import sequtils, strutils, sugar
import mnist_tools, simpleneuralnets

proc main() =
    # Prepare training data
    let trainingData = mnistTrainingData()
    let (trainingLabels, trainingImages) = unzip(trainingData)
    # Normalize grayscale values
    let normalizedTrainingData = trainingImages.map(proc(image: seq[
            int]): seq[
        float] = image / 255.0)

    # Prepare test data
    let testData = mnistTestData()
    let (testLabels, testImages) = unzip(testData)
    let normalizedTestData = testImages.map(proc(image: seq[int]): seq[
            float] = image / 255.0)

    # Hyperparameters
    let learningRate = 0.5
    let epochs = 50
    let hiddenLayersNumber = 1
    let hiddenLayerNodes = 20
    let outputLayerNumber = 10
    let inputLayerNumber = 784

    let networkShape = @[inputLayerNumber, hiddenLayerNodes,
            outputLayerNumber]

    var network = newNetwork([784, 20, 10])
    # echo("Network stats -> ", "# of layers: ", network.len,
    #         "# of nodes per layer: ", network[0][0].weights.len, " ",
    #                 network[0].len, " ", network[1].len)

    trainBNN(network, normalizedTrainingData, trainingLabels, learningRate,
            epochs)

    # Accuracy on training data
    let trainingComputed = lc[singleOutcome(classify(network,
            normalizedTrainingData[
            i])) | (i <- 0..<normalizedTrainingData.len), int]
    echo("Expected")

    let accuracy = computeAccuracy(trainingLabels, trainingComputed)
    echo("Accuracy: ", accuracy)

    let confMatrix = confusionMatrix(trainingLabels, trainingComputed, 11)
    echo(confMatrix.join("\n"))

    # Accuracy on test data
    echo("TESTING DATA")
    let testingComputed = lc[singleOutcome(classify(network,
            normalizedTestData[
            i])) | (i <- 0..<normalizedTestData.len), int]

    let testingAccuracy = computeAccuracy(testLabels, testingComputed)
    echo("Accuracy: ", testingAccuracy)

    let testingConfMatrix = confusionMatrix(testLabels, testingComputed, 11)
    echo(testingConfMatrix.join("\n"))


# Accuracy on test data


when isMainModule:
    main()
