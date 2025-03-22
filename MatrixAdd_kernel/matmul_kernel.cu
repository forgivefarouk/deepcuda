

__global__ void matmul_kernel(float *A , float *B , float *C , int ar , int ac , int br , int bc){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (row < ar && col < bc){
      float sum = 0;
      for (int i = 0 ; i < ac ; i++){
        sum += A[row * ac + i] * B[i * bc + col];
      }
      C[row * bc + col] = sum;
    }
  
  }
  
  void matmul_gpu(float *A , float *B , float *C , int ar , int ac , int br , int bc){
  
    // make cuda memory
    float *A_d , *B_d , *C_d;
    cudaMalloc((void **)&A_d , ar * ac * sizeof(float));
    cudaMalloc((void **)&B_d , br * bc * sizeof(float));
    cudaMalloc((void **)&C_d , ar * bc * sizeof(float));
  
    // copy host memory to device memory
    cudaMemcpy(A_d , A , sizeof(float) * ar * ac , cudaMemcpyHostToDevice);
    cudaMemcpy(B_d , B , sizeof(float) * br * bc , cudaMemcpyHostToDevice);
  
    // init C_d to zero
    cudaMemset(C_d , 0 , sizeof(float) * ar * bc);
  
    // lunch grid kernel
    dim3 N_THREADS_PER_BLOCK(32 , 32);
    dim3 N_BLOCKS_PER_GRID((bc + N_THREADS_PER_BLOCK.x -1) / N_THREADS_PER_BLOCK.x , (ar + N_THREADS_PER_BLOCK.y -1) / N_THREADS_PER_BLOCK.y);
    matmul_kernel<<<N_BLOCKS_PER_GRID , N_THREADS_PER_BLOCK>>>(A_d , B_d , C_d , ar , ac , br , bc);
  
    // copy device memory to host memory
    cudaMemcpy(C , C_d , sizeof(float) * ar * bc , cudaMemcpyDeviceToHost);
    // free cuda memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
  }
  
  
  int main() {
    // Example matrix dimensions
    int ar = 1024, ac = 512;
    int br = 512, bc = 768;
    
    
    // Allocate host memory
    float *A = (float*)malloc(sizeof(float) * ar * ac);
    float *B = (float*)malloc(sizeof(float) * br * bc);
    float *C = (float*)malloc(sizeof(float) * ar * bc);
    
    // Initialize matrices (example)
    for (int i = 0; i < ar * ac; i++) A[i] = 1.0f;
    for (int i = 0; i < br * bc; i++) B[i] = 2.0f;
    
    // Call GPU matrix multiplication
    matmul_gpu(A, B, C, ar, ac, br, bc);
    
  
    
    // Free host memory
    free(A);
    free(B);
    free(C);
    
    return 0;
  }