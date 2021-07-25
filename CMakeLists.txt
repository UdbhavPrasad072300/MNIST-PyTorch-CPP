cmake_minimum_required (VERSION 3.8)
project("MNIST_PyTorch_CPP")

set(CMAKE_PREFIX_PATH /home/udbhavprasad/MNIST-PyTorch-CPP/libtorch /home/udbhavprasad/build)
set(CMAKE_INSTALL_PREFIX /usr/local)
set(CMAKE_BUILD_TYPE Release)
set(OPENCV_EXTRA_MODULES_PATH /home/udbhavprasad/opencv-master/modules)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (${PROJECT_NAME} "main.cpp"
                                include/network.h
                                src/network.cpp)



include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_range_for)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

message(STATUS "Torch library status:")
message(STATUS "    version: ${TORCH_VERSION}")
message(STATUS "    config: ${TORCH_DIR}")
message(STATUS "    libraries: ${TORCH_LIBS}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
