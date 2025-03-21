#include <iostream>

__global__ void vecadd_kernel(float * x , float * y , float *z , unsigned int N){

  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < N){
    z[index] = x[index] + y[index];
  }
}

void vecadd_gpu(float * x , float * y , float *z ,unsigned int  N){
  // allocate memory on device
  float * x_d , *y_d , *z_d ;
  cudaMalloc((void **)&x_d , N * sizeof(float));
  cudaMalloc((void **)&y_d , N * sizeof(float));
  cudaMalloc((void **)&z_d , N * sizeof(float));

  // transfer data from host to device
  cudaMemcpy(x_d , x , N*sizeof(float) , cudaMemcpyHostToDevice);
  cudaMemcpy(y_d , y , N*sizeof(float) , cudaMemcpyHostToDevice);

  // launch kernel
  int N_THREADS_PER_BLOCK = 512 ;
  // ceil(N/N_THREADS_PER_BLOCK) ;
  int N_BLOCKS = (N+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK ;
  vecadd_kernel<<<N_BLOCKS , N_THREADS_PER_BLOCK>>>(x_d , y_d , z_d , N);

  // transfer data from device to host
  cudaMemcpy(z , z_d , N*sizeof(float) , cudaMemcpyDeviceToHost);


  // free memory from device
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
}


int main(){
  const int N = 100000000;
  float *x , *y , *z ;
  x = (float *)malloc(N*sizeof(float));
  y = (float *)malloc(N*sizeof(float));
  z = (float *)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(2 * i);
    }

  vecadd_gpu(x , y , z , N);

  free(x);
  free(y);
  free(z);

  return 0;

}