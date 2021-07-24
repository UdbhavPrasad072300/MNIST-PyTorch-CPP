cmake_minimum_required (VERSION 3.8)

project("MNIST_PyTorch_CPP")

set(CMAKE_PREFIX_PATH /home/udbhavprasad/MNIST-PyTorch-CPP/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
