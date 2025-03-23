%%writefile blur_img.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define BLUR_SIZE 5

__global__ void blur_kernel(const unsigned char *image, unsigned char *blur, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        // Process each color channel independently
        for (int c = 0; c < channels; c++) {
            unsigned int sum = 0;
            int count = 0;
            // Loop over the neighborhood with boundary checking
            for (int i = row - BLUR_SIZE; i <= row + BLUR_SIZE; i++) {
                for (int j = col - BLUR_SIZE; j <= col + BLUR_SIZE; j++) {
                    if (i >= 0 && i < height && j >= 0 && j < width) {
                        int idx = (i * width + j) * channels + c;
                        sum += image[idx];
                        count++;
                    }
                }
            }
            // Write the average value to the output pixel for the current channel
            blur[(row * width + col) * channels + c] = (unsigned char)(sum / count);
        }
    }
}

void blur_image_gpu(unsigned char *img, unsigned int width, unsigned int height, int channels) {
    unsigned char *img_d, *blur_d;
    size_t size = width * height * channels * sizeof(unsigned char);
    unsigned char *blur = (unsigned char*)malloc(size);
    
    // Allocate device memory
    cudaMalloc((void **)&img_d, size);
    cudaMalloc((void **)&blur_d, size);
    
    // Copy image data to the device
    cudaMemcpy(img_d, img, size, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 blocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the blur kernel
    blur_kernel<<<blocks, threadsPerBlock>>>(img_d, blur_d, width, height, channels);
    
    // Check for any kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
    }
    
    cudaDeviceSynchronize();
    
    // Copy the blurred image back to the host
    cudaMemcpy(blur, blur_d, size, cudaMemcpyDeviceToHost);
    
    // Save the blurred image (3 channels)
    stbi_write_jpg("dog_blur.jpg", width, height, channels, blur, 90);
    printf("Blurred image saved as dog_blur.jpg\n");
    
    // Free device memory
    cudaFree(img_d);
    cudaFree(blur_d);
    
    // Free host memory for the blurred image
    free(blur);
}

int main() {
    const char* image_path = "./dog.jpg";
    
    int width, height, channels;
    
    printf("Loading image: %s\n", image_path);
    // Load the image as a color image (3 channels)
    unsigned char* img_data = stbi_load(image_path, &width, &height, &channels, 3);
    if (!img_data) {
        printf("Error loading image\n");
        return 1;
    }
    
    channels = 3;
    blur_image_gpu(img_data, width, height, channels);
    
    stbi_image_free(img_data);
    return 0;
}
