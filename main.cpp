#include <torch/torch.h>
#include <iostream>


// Hyperparameters

const int64_t INPUT_SIZE = 784;
const int64_t NUM_CLASSES = 10;
const int64_t BATCH_SIZE = 1000;
const size_t NUM_EPOCHS = 10;
const double LR = 0.001;

const std::string MNIST_data_path = "./data/mnist/";

// Model

struct Net : torch::nn::Module {
	Net(int64_t N, int64_t M) {
		W = register_parameter("W", torch::randn({ N, M }));
		b = register_parameter("b", torch::randn(M));
	}
	torch::Tensor forward(torch::Tensor input) {
		return torch::addmm(b, input, W);
	}
	torch::Tensor W, b;
};


int main() {
	std::cout << "MNIST Classifier\n" << std::endl;
	// DEVICE
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "GPU being Used" : "CPU being Used") << '\n' << std::endl;

	torch::Tensor tensor = torch::rand({ 2, 3 }, torch::kCUDA);
	std::cout << tensor << std::endl;
}
