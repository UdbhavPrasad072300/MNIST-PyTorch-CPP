#include <torch/torch.h>
#include <iostream>
#include "include/network.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// Hyperparameters

const int64_t INPUT_SIZE = 784;
const ino64_t HIDDEN_SIZE = 128;
const int64_t NUM_CLASSES = 10;
const int64_t BATCH_SIZE = 1000;
const size_t NUM_EPOCHS = 10;
const double LR = 0.001;

const std::string MNIST_data_path = "/home/udbhavprasad/MNIST-PyTorch-CPP/mnist";

int main() {
	std::cout << "MNIST Classifier\n" << std::endl;

	// DEVICE

	auto cuda_available = torch::cuda::is_available();
	torch::Device DEVICE(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "GPU being Used" : "CPU being Used") << '\n' << std::endl;

	// Dataset

    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    std::cout << "Train Dataset is of size: " << train_dataset.size().value() << std::endl;
    std::cout << "Test Dataset is of size: " << test_dataset.size().value() << std::endl;

    // Dataloader

    auto train_loader = torch::data::make_data_loader(std::move(train_dataset), BATCH_SIZE);
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), BATCH_SIZE);

	// Model

    Net model(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES);
    model.to(DEVICE);

    std::cout << model << std::endl;

    // Test Run

	torch::Tensor tensor = torch::rand({ 2, 3 }, torch::kCUDA);
	std::cout << tensor << std::endl;

	std::cout << "Program Finished" << std::endl;
}
