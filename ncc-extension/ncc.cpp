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

  X.set_requires_grad(false);
  Y.set_requires_grad(false);

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

  //X_pad.set_requires_grad(false);
  //Y_pad.set_requires_grad(false);

  //cout << "Sizes after padding are: ";
  //cout << X_pad.sizes();
  //cout << Y_pad.sizes();


  torch::Tensor output = torch::empty({in_depth,in_height,in_width, sample_size, in_width*patch_size });  //25*37*12* batch_size * 60
  //auto out_access = output.accessor<float,5>();

  output.set_requires_grad(false);

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
              ncc_vector.reserve(patch_size*in_width);

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

              stacked.set_requires_grad(false);

              //add result to position x,y w.r.t. to current depth, in output tensor

              output[i][j][k] = stacked;
          }


      }
  } // for all depths

  //output at this point, is 25 x 37 x 12 x batch_size x 60

  return output.reshape({sample_size, output.size(-1)*in_depth, in_height, in_width});
}

// Using NCC gradient formula as defined in the original paper

std::vector<torch::Tensor> ncc_backward(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor grad_out,
    int patch_size = 5,
    int stride =1,
    float epsilon = 0.01) {




  int sample_size = X.size(0);
  int in_depth = X.size(1);
  int in_height= X.size(2);
  int in_width= X.size(3);

  int d = patch_size/2;

  torch::Tensor gradX = torch::zeros({in_depth, in_height, in_width, sample_size });
  //torch::Tensor gradX_ = gradX.unsqueeze_(-1);

  torch::Tensor gradY = torch::zeros({in_depth, in_height, in_width, sample_size });
  //torch::Tensor gradY_ = gradY.unsqueeze_(-1);

  cout<< gradX.sizes();

  auto X_pad = at::constant_pad_nd(X, {d, d, d, d}, /*pad with value:*/ 0);
  auto Y_pad = at::constant_pad_nd(X, {d, d, 2 * d, 2 * d}, /*pad with value:*/ 0);

  //It is 15, 1500, 37, 12   so reshape it first to revert operation backwards
  torch::Tensor grad_out_r = grad_out.reshape({in_depth,in_height,in_width, sample_size, in_width*patch_size, 1}); //25*37*12* batch_size * 60


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
      // Added pre-computed components for gradients as well

      // Right component of grad input 1
      auto E_norm_d_right = (Esr - E_means)/((Esr.size(-1)-1)* E_stdr.pow(2));

      // The usual one needed for NCC computation
      auto E_norm = E_norm_d_right * E_stdr;

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
      // Added pre-computed components for gradients as well
      // Right component of grad input 2
      auto F_norm_d_right = (Fsr - F_means)/ ((Fsr.size(-1)-1)* F_stdr.pow(2));

      // The usual one needed for NCC computation
      auto F_norm = F_norm_d_right * (Fsr.size(-1)-1)* F_stdr;

      /*
      // Note that E_stdr and F_strd have different sizes due to padding,
      // so we need to first pad E_stdr with replicated values for 2 up and 2 down
      // Replication 2d pad is implemented only for the last 2 dims of 4d tensors
      //As per the docs, so we need to inconveniently reshape E_stdr and back again
      auto E_stdr_2 = at::replication_pad2d(E_stdr.view({sample_size, in_width, in_height, 1}),{0,0,2,2});
      auto E_stdr_2_ = E_stdr_2.view({E_stdr_2.size(0),E_stdr_2.size(2), E_stdr_2.size(1), E_stdr_2.size(3)});

      //similarly, for E_norm
      auto E_norm_2 = at::replication_pad2d(E_norm.view({sample_size, in_width, in_height, E_norm.size(3)}),{0,0,2,2});
      auto E_norm_2_ = E_norm_2.view({E_norm_2.size(0),E_norm_2.size(2), E_norm_2.size(1), E_norm_2.size(3)});
      */

      //cout<<E_stdr_bottom.sizes();

      /*
      - Two nested loops pixel by pixel in E_norm just for
      dot product and sum
      */

      //cout << "Fnorm size: ";
      //cout << E_norm_d_right.sizes();
      //cout << E_norm_d_left.sizes();
      //cout << F_norm_d_right.sizes();
      //cout << F_norm_d_left.sizes();


      for(int j=0; j<E_norm.size(1);j++){   //37 times

          //cout << j;


          for(int k=0; k<E_norm.size(2);k++){ //12 times

              //For each pixel in E_norm, i.e., (x,y) coord

              auto E = E_norm.select(/*dim=*/1, /*index=*/j).select(/*dim=*/1, /*index=*/k);

              auto E_d_right = E_norm_d_right.select(/*dim=*/1, /*index=*/j).select(/*dim=*/1, /*index=*/k); //E_norm_d_right @ E

              auto E_std_cross = E_stdr.select(/*dim=*/1, /*index=*/j).select(/*dim=*/1, /*index=*/k);  // Specific std for left components

              //cout << "Single E size: ";
              //cout << E.sizes();

              std::vector<torch::Tensor> dE_vector;
              dE_vector.reserve(patch_size*in_width);

              std::vector<torch::Tensor> dF_vector;
              dF_vector.reserve(patch_size*in_width);

              for(int y=j; y<j+5; y++){  //5 times


                      for(int n=0; n<in_width;n++){ //12 times

                          //cout << "9";


                          auto  F= F_norm.select(/*dim=*/1, /*index=*/y).select(/*dim=*/1, /*index=*/n);

                          //cout << "10";
                          auto  F_d_right= F_norm_d_right.select(/*dim=*/1, /*index=*/y).select(/*dim=*/1, /*index=*/n); //E_norm_d_right @ F
                          //cout << "11";
                          auto F_std_cross = F_stdr.select(/*dim=*/1, /*index=*/y).select(/*dim=*/1, /*index=*/n);    // Specific std for left components
                          //cout << "12";
                          // Left component of grad input 1
                          auto F_d_left = (F_d_right * F_std_cross)/ E_std_cross ;
                          //cout << "13";
                          // Left component of grad input 2
                          auto E_d_left = E / F_std_cross;
                          //cout << "14";
                          auto ncc_values = at::sum(E * F, 1, true);

                          // Derivative of norm cross corr w.r.t. Ei
                          //cout << "15";


                          auto d_ncc_E_right = E_d_right*ncc_values;
                          auto d_ncc_F_right = F_d_right*ncc_values;

                          //cout << d_ncc_E_right.sizes();
                          //cout << "16";
                          auto d_ncc_E = F_d_left - d_ncc_E_right ;
                          auto d_ncc_F = E_d_left - d_ncc_F_right ;


                          // Derivative of norm cross corr w.r.t. Fi
                          //cout << "17";
                          dE_vector.push_back(d_ncc_E);
                          dF_vector.push_back(d_ncc_F);

                      }

              }


              torch::Tensor Estacked = at::stack(dE_vector, 1); //batch_size x 60, where 60 = 12*5
              torch::Tensor Fstacked = at::stack(dF_vector, 1);

              Estacked.set_requires_grad(false);
              Fstacked.set_requires_grad(false);

              //cout<<Estacked.sizes();
              //cout<<grad_out_r[i][j][k].sizes();

              //add result to position x,y w.r.t. to current depth, in output tensor
              auto curr_gradout_E = grad_out_r[i][j][k]*Estacked ; //= stacked;
              auto curr_gradout_F = grad_out_r[i][j][k]*Fstacked  ;

              // Summing gradients over all dimensions at that X,Y
              //Not keeping positional info stacked as in forward
              //Needed to push back to two pipelines before in the Net
              auto gradout_E = at::sum(at::sum(curr_gradout_E,1),1);
              auto gradout_F = at::sum(at::sum(curr_gradout_F,1),1);


              //gradX[i][j][k]= gradout_E;
              //gradY[i][j][k]= gradout_F;
          }


      }

  } // for all depths


  return {gradX.reshape(X.sizes()), gradY.reshape(Y.sizes())};

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ncc_forward, "NCC forward");
  m.def("backward", &ncc_backward, "NCC backward");
}