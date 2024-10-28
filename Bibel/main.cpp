#include "NeuralNetwork.hpp"
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

std::unordered_map<std::string, int> build_vocabulary(const std::string& filename) {
    std::unordered_map<std::string, int> word_to_index;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return word_to_index;
    }

    std::string word;
    int index = 0;
    while (file >> word) {
        if (word_to_index.find(word) == word_to_index.end()) {
            word_to_index[word] = index++;
        }
    }
    return word_to_index;
}

void trainBatch(NeuralNetwork& nn, const std::vector<std::pair<std::vector<double>, std::vector<double>>>& batch_data, double learningRate) {
    for (const auto& data : batch_data) {
        train(nn, data.first, data.second, learningRate);  // Train without mutex or threading
    }
}

void generate_text(const std::string& start_word, NeuralNetwork& nn,
    const std::unordered_map<std::string, int>& word_to_index,
    const std::vector<std::string>& index_to_word, int num_words) {
    std::ofstream file("Story.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file 'Story.txt'" << std::endl;
        return;
    }

    std::string current_word = start_word;
    if (word_to_index.find(current_word) == word_to_index.end()) {
        std::cerr << "Error: Start word not found in vocabulary!" << std::endl;
        return;
    }

    file << current_word << " ";
    for (int i = 0; i < num_words - 1; ++i) {
        std::vector<double> input_vector(word_to_index.size(), 0.0);
        input_vector[word_to_index.at(current_word)] = 1.0;

        std::vector<double> output_vector = nn.predict(input_vector);
        size_t max_index = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));

        if (max_index < index_to_word.size()) {
            std::string next_word = index_to_word[max_index];
            file << next_word << " ";
            current_word = next_word;
        }
        else {
            std::cerr << "Error: Predicted index out of bounds!" << std::endl;
            break; // Exit if max_index is out of bounds
        }
    }
    file << std::endl;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    std::string filename = "bible.txt";
    std::unordered_map<std::string, int> word_to_index = build_vocabulary(filename);
    if (word_to_index.empty()) {
        std::cerr << "Error: Vocabulary could not be built. Exiting." << std::endl;
        return 1;
    }

    std::vector<std::string> index_to_word(word_to_index.size());
    for (const auto& pair : word_to_index) {
        index_to_word[pair.second] = pair.first;
    }

    int input_size = word_to_index.size();
    int hidden_size = word_to_index.size();
    int output_size = input_size;
    NeuralNetwork nn(input_size, hidden_size, output_size);

    int batchSize = (int)pow(2,3);  // Reduced batch size for faster processing
    double learningRate = 0.01;
    int epochs = 100;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Starting epoch " << epoch + 1 << " of " << epochs << std::endl;

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for reading." << std::endl;
            return 1;
        }

        std::string prev_word, word;
        file >> prev_word;  // Read the first word
        std::vector<std::pair<std::vector<double>, std::vector<double>>> batch_data;

        int current = 1;

        while (file >> word) {
            if (word_to_index.find(prev_word) != word_to_index.end() && word_to_index.find(word) != word_to_index.end()) {
                std::vector<double> input(input_size, 0.0);
                std::vector<double> output(input_size, 0.0);
                input[word_to_index[prev_word]] = 1.0;
                output[word_to_index[word]] = 1.0;
                batch_data.emplace_back(input, output);

                // Check if the batch size is reached
                if (batch_data.size() >= batchSize) {
                    std::cout << "Training batch " << current << " of epoch " << epoch + 1 << std::endl;
                    trainBatch(nn, batch_data, learningRate);  // Train with the current batch
                    batch_data.clear();  // Clear the batch for the next set of data
                    current++;
                }
            }
            prev_word = word;  // Update the previous word
        }

        // Final training if there are remaining data
        if (!batch_data.empty()) {
            trainBatch(nn, batch_data, learningRate);
        }

        file.close();  // Close the file after each epoch
        std::cout << "Completed epoch " << epoch + 1 << " of " << epochs << std::endl;
    }

    std::string start_word = "The";
    int num_words_to_generate = 500;
    generate_text(start_word, nn, word_to_index, index_to_word, num_words_to_generate);

    std::cout << "Text generated." << std::endl;

    return 0;
}