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

    train_size = train_dataset.size().value();
    test_size = test_dataset.size().value();

    std::cout << "Train Dataset is of size: " << train_size << std::endl;
    std::cout << "Test Dataset is of size: " << test_size << std::endl;

    // Dataloader

    auto train_loader = torch::data::make_data_loader(std::move(train_dataset), BATCH_SIZE);
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), BATCH_SIZE);

	// Model

    Net model(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES);
    model.to(DEVICE);

    std::cout << model << std::endl;

    // Optimizer

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(LR));

    // Training

    for (size_t epoch=0; epoch < NUM_EPOCHS; epoch++) {

        for (auto& batch : *train_loader) {
            auto data = batch.data.view({BATCH_SIZE, -1}).to(DEVICE);
            auto target = batch.target.to(DEVICE);

            auto out = model.forward(data);

            auto loss = torch::nn::functional::cross_entropy(out, target);

            auto prediction = output.argmax(1);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // std::cout << data << std::endl;
        }

        std::cout << "Epoch " << epoch << std::endl;
    }

    // Test Run

    model.eval();
    float test_loss = 0
    size_t num_correct = 0;

    for (auto& batch : *test_loader) {
        auto data = batch.data.view({ BATCH_SIZE, -1 }).to(DEVICE);
        auto target = batch.target.to(DEVICE);

        auto out = model.forward(data);

        auto loss = torch::nn::functional::cross_entropy(out, target);

        test_loss += loss.item<double>() * data.size(0);
        
        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    auto test_accuracy = static_cast<double>(num_correct) / test_sizes;
    auto test_loss = test_loss / test_size;

    std::cout << "Test Accuracy: " << test_accuracy << std::endl;
    std::cout << "Test Loss" << test_loss << std::endl;


    // Example

	torch::Tensor tensor = torch::rand({ 2, 3 }, torch::kCUDA);
	std::cout << tensor << std::endl;

	std::cout << "Program Finished" << std::endl;
}
