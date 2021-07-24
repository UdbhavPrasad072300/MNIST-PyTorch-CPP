#include <torch/torch.h>
#include <iostream>
#include "include/network.h"


// Hyperparameters

const int64_t INPUT_SIZE = 784;
const ino64_t HIDDEN_SIZE = 128;
const int64_t NUM_CLASSES = 10;
const int64_t BATCH_SIZE = 1000;
const size_t NUM_EPOCHS = 10;
const double LR = 0.001;

const std::string MNIST_data_path = "./data/";

int main() {
	std::cout << "MNIST Classifier\n" << std::endl;

	// DEVICE

	auto cuda_available = torch::cuda::is_available();
	torch::Device DEVICE(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "GPU being Used" : "CPU being Used") << '\n' << std::endl;

	// Model

    Net model(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES);
    model.to(DEVICE);

    std::cout << model << std::endl;

    // Test Run

	torch::Tensor tensor = torch::rand({ 2, 3 }, torch::kCUDA);
	std::cout << tensor << std::endl;

	std::cout << "Program Finished" << std::endl;
}
