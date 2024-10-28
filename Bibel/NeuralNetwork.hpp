#pragma once

#include <vector>
#include <cstdlib>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <thread>

class Layer {
public:
    std::vector<double> neurons;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(int numNeurons, int numInputs) {
        neurons.resize(numNeurons);
        biases.resize(numNeurons);
        weights.resize(numNeurons, std::vector<double>(numInputs));

        for (int i = 0; i < numNeurons; i++) {
            biases[i] = ((double)rand() / RAND_MAX) - 0.5;
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }
};

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double relu(double x) {
    return std::max(0.0, x);
}

double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

std::vector<double> forwardPass(const std::vector<double>& inputs, Layer& layer) {
    std::vector<double> outputs(layer.neurons.size());
    for (size_t i = 0; i < layer.neurons.size(); i++) {
        double activation = layer.biases[i];
        activation += std::inner_product(inputs.begin(), inputs.end(), layer.weights[i].begin(), 0.0);
        outputs[i] = relu(activation); // Use ReLU here
    }
    return outputs;
}

// And update the training function to use reluDerivative

class NeuralNetwork {
public:
    Layer hiddenLayer;
    Layer outputLayer;

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) :
        hiddenLayer(hiddenSize, inputSize), outputLayer(outputSize, hiddenSize) {}

    std::vector<double> predict(const std::vector<double>& inputs) {
        std::vector<double> hiddenOutput = forwardPass(inputs, hiddenLayer);
        return forwardPass(hiddenOutput, outputLayer);
    }
};

void train(NeuralNetwork& nn, const std::vector<double>& inputs, const std::vector<double>& expectedOutput, double learningRate) {
    std::vector<double> hiddenOutput = forwardPass(inputs, nn.hiddenLayer);
    std::vector<double> finalOutput = forwardPass(hiddenOutput, nn.outputLayer);

    std::vector<double> outputDeltas(nn.outputLayer.neurons.size());
    for (size_t i = 0; i < outputDeltas.size(); i++) {
        double error = expectedOutput[i] - finalOutput[i];
        outputDeltas[i] = error * reluDerivative(finalOutput[i]); // Use ReLU derivative
    }

    std::vector<double> hiddenDeltas(nn.hiddenLayer.neurons.size());
    for (size_t i = 0; i < hiddenDeltas.size(); i++) {
        double error = 0.0;
        for (size_t j = 0; j < outputDeltas.size(); j++) {
            error += outputDeltas[j] * nn.outputLayer.weights[j][i];
        }
        hiddenDeltas[i] = error * reluDerivative(hiddenOutput[i]); // Use ReLU derivative
    }

    // Update output layer weights and biases
    for (size_t i = 0; i < nn.outputLayer.neurons.size(); i++) {
        nn.outputLayer.biases[i] += outputDeltas[i] * learningRate;
        for (size_t j = 0; j < nn.hiddenLayer.neurons.size(); j++) {
            nn.outputLayer.weights[i][j] += hiddenOutput[j] * outputDeltas[i] * learningRate;
        }
    }

    // Update hidden layer weights and biases
    for (size_t i = 0; i < nn.hiddenLayer.neurons.size(); i++) {
        nn.hiddenLayer.biases[i] += hiddenDeltas[i] * learningRate;
        for (size_t j = 0; j < inputs.size(); j++) {
            nn.hiddenLayer.weights[i][j] += inputs[j] * hiddenDeltas[i] * learningRate;
        }
    }
}

void trainOnBatch(NeuralNetwork& nn, const std::vector<std::pair<std::vector<double>, std::vector<double>>>& batch_data, double learningRate) {
    for (const auto& data : batch_data) {
        train(nn, data.first, data.second, learningRate);
    }
}



void trainNetwork(NeuralNetwork& nn, const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainingData, int epochs, double learningRate, int batchSize) {
    size_t dataSize = trainingData.size();

    for (int e = 0; e < epochs; e++) {
        std::cout << "Epoch " << e + 1 << "/" << epochs << std::endl;

        // Shuffle training data before each epoch (optional but recommended)
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<std::pair<std::vector<double>, std::vector<double>>> shuffledData = trainingData;
        std::shuffle(shuffledData.begin(), shuffledData.end(), g);

        for (size_t i = 0; i < dataSize; i += batchSize) {
            // Create a mini-batch
            std::vector<std::pair<std::vector<double>, std::vector<double>>> batch(shuffledData.begin() + i,
                shuffledData.begin() + std::min(i + batchSize, dataSize));

            // Launch a thread to process this batch
            std::thread t(trainOnBatch, std::ref(nn), batch, learningRate);
            t.detach(); // Detach the thread for concurrent execution
        }

        // Optional: wait for all threads to complete (not necessary if using detached threads)
        // You would need to implement a way to keep track of them, e.g., using std::promise and std::future.
    }
}
