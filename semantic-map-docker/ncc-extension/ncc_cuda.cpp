// Agnese Chiatti 2019-02-09
// ncc.cpp
// C++ extension for Pytorch to compute Normalized Cross Correlation
// of 2 input feature maps, as illustrated by Submariam et al. (NIPS 2016)

#include <torch/extension.h>

#include <iostream>
#include <fstream>
using namespace std;

#include <vector>

torch::Tensor ncc_cuda_forward(
    torch::Tensor X,
    torch::Tensor Y,
    int patch_size = 5,
    int stride =1,
    float epsilon = 0.01);

torch::Tensor ncc_cuda_backward(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor grad_out,
    int patch_size = 5,
    int stride =1,
    float epsilon = 0.01) ;


// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor ncc_forward(
    torch::Tensor X,
    torch::Tensor Y,
    int patch_size = 5,
    int stride =1,
    float epsilon = 0.01) {

  CHECK_INPUT(X);
  CHECK_INPUT(Y);
  
  return ncc_cuda_forward(X,Y,patch_size,stride,epsilon);
}


// Using NCC gradient formula as defined in the original paper

std::vector<torch::Tensor> ncc_backward(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor grad_out,
    int patch_size = 5,
    int stride =1,
    float epsilon = 0.01) {


  CHECK_INPUT(X);
  CHECK_INPUT(Y);
  CHECK_INPUT(grad_out);
  
  return ncc_cuda_backward(X,Y,grad_out, patch_size,stride,epsilon);
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ncc_forward, "NCC forward (CUDA)");
  m.def("backward", &ncc_backward, "NCC backward (CUDA)");
}
