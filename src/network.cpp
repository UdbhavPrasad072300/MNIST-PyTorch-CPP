#include "../include/network.h"
#include <torch/torch.h>


Net::Net(int64_t input_size, int64_t hidden_size, int64_t num_classes)
        : fc1(input_size, hidden_size), fc2(hidden_size, num_classes) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor Net::forward(torch::Tensor x) {
    x = torch::nn::functional::relu(fc1->forward(x));
    return fc2->forward(x);
}
