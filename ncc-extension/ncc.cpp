// Agnese Chiatti 2019-02-09
// ncc.cpp
// C++ extension for Pytorch to compute Normalized Cross Correlation
// of 2 input feature maps, as illustrated by Submariam et al. (NIPS 2016)

#include <torch/extension.h>

#include <iostream>
#include <fstream>
using namespace std;

#include <vector>

torch::Tensor ncc_forward(
    torch::Tensor X,
    torch::Tensor Y,
    int patch_size = 5,
    int stride =1,
    float epsilon = 0.01) {

  // copies for padding

  //cout << "Input sizes are: ";
  //cout << X.sizes();
  //cout << Y.sizes();

  int sample_size = X.size(0);
  int in_depth = X.size(1);
  int in_height= X.size(2);
  int in_width= X.size(3);

  //int out_depth= patch_size*in_width*in_depth;

  int d = patch_size/2;

  /*
        for each depth i in range(25):

            1. take the ith 37x12 feature map from X
                1.a create copy of X with padding of two at each margin -> 41x16
            2. take the ith 37x12 feature map from Y
                2.a create copy of Y with same size of 1.a, but extra vertical padding of two each -> 45x16
*/


  auto X_pad = at::constant_pad_nd(X, {d, d, d, d}, 0);
  auto Y_pad = at::constant_pad_nd(X, {d, d, 2 * d, 2 * d}, 0);


  //cout << "Sizes after padding are: ";
  //cout << X_pad.sizes();
  //cout << Y_pad.sizes();


  torch::Tensor output = torch::empty({in_depth,in_height,in_width, sample_size, in_width*patch_size });  //25*37*12* batch_size * 60
  //auto out_access = output.accessor<float,5>();

  for (int i=0; i< in_depth; i++){

      auto X_i = X_pad.select(/*dim=*/1, /*index=*/i);
      auto Y_i = Y_pad.select(/*dim=*/1, /*index=*/i);

      //cout << "After slicing on just one depth: ";
      //cout << X_i.sizes();
      //cout << Y_i.sizes();

      auto Es = X_i.unfold( /*dim=*/ 1, /*size=*/ patch_size, /*step=*/ stride).unfold(/*dim=*/ 2, /*size=*/ patch_size, /*step=*/ stride);

      //cout << "Find 5x5 patches: ";
      //cout << Es.sizes();

      auto Esr = Es.reshape({Es.size(0),Es.size(1), Es.size(2), patch_size*patch_size}); //reshaping neighbourhoods as 25x25


      auto E_means = Esr.mean(/*dim=*/-1, /*keepdim=*/true);

      auto E_std = Esr.std(/*dim=*/-1, /*keepdim=*/true) + epsilon;

      auto E_stdr = E_std.reshape(E_means.sizes());


      // Normalize all E matrices
      auto E_norm = (Esr - E_means)/((Esr.size(-1)-1)* E_stdr);



      /*
      - Compute all possible rectangles in Y_i
      and normalize them
      */

      //cout << "Y_i size: ";
      //cout << Y_i.sizes();

      auto Fs = Y_i.unfold( /*dim=*/ 1, /*size=*/ patch_size, /*step=*/ stride).unfold(/*dim=*/ 2, /*size=*/ patch_size, /*step=*/ stride);
      //cout << "Fs size: ";
      //cout << Fs.sizes();
      auto Fsr = Fs.reshape({Fs.size(0),Fs.size(1), Fs.size(2), patch_size*patch_size}); //reshaping neighbourhoods as 25x1


      auto F_means = Fsr.mean(/*dim=*/-1, /*keepdim=*/true);
      auto F_std = Fsr.std(/*dim=*/-1, /*keepdim=*/true) + epsilon;
      auto F_stdr = F_std.reshape(F_means.sizes());

      // Normalize all E matrices
      auto F_norm = (Fsr - F_means)/ F_stdr;

      /*
      - Two nested loops pixel by pixel in E_norm just for
      dot product and sum
      */

      //cout << "Fnorm size: ";
      //cout << F_norm.sizes();


      for(int j=0; j<E_norm.size(1);j++){   //37 times

          //cout << j;

          for(int k=0; k<E_norm.size(2);k++){ //12 times

              //For each pixel in E_norm, i.e., (x,y) coord

              auto E = E_norm.select(/*dim=*/1, /*index=*/j).select(/*dim=*/1, /*index=*/k);

              //cout << "Single E size: ";
              //cout << E.sizes();

              std::vector<torch::Tensor> ncc_vector;


              for(int y=j; y<j+5; y++){  //5 times


                      for(int n=0; n<in_width;n++){ //12 times


                          auto  F= F_norm.select(/*dim=*/1, /*index=*/y).select(/*dim=*/1, /*index=*/n);

                          //cout << "Single F size: ";
                          //cout << F.sizes();

                          //At this point, E and F are both matrices of shape: batch_sizex25

                          // E*F yields batch_size x 25 (it is done element-wise)

                          //cout << "Ncc size: ";
                          //cout << at::sum(E * F, 1).sizes();

                          ncc_vector.push_back(at::sum(E * F, 1)); //summing over all 25 dims , obtaining batch_size scalars


                      }

              }


              torch::Tensor stacked = at::stack(ncc_vector, 1); //batch_size x 60, where 60 = 12*5

              //add result to position x,y w.r.t. to current depth, in output tensor

              output[i][j][k] = stacked;
          }


      }
  } // for all depths

  //output at this point, is 25 x 37 x 12 x batch_size x 60

  return output.reshape({sample_size, output.size(-1)*in_depth, in_height, in_width});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ncc_forward, "NCC forward");
}