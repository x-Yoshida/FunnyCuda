cmake_minimum_required(VERSION 3.8)
project(cuda_test LANGUAGES CXX CUDA)

# enable_language(CU)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(cuda_test src/cuda_test.cu)

target_link_libraries(cuda_test ${OpenCV_LIBS})
