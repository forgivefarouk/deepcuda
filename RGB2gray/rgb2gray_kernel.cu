#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void rgb2gray_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < height && col < width){
        int index = row * width + col;
        // Using weighted formula for better grayscale conversion (standard RGB to luminance)
        gray[index] = (unsigned char)(0.3f * red[index] + 0.6f * green[index] + 0.1f * blue[index]);
    }
}


void rgb2gray_gpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned int width, unsigned int height){
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    unsigned char *gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    

    cudaMalloc((void **)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&gray_d, width * height * sizeof(unsigned char));
    
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMemset(gray_d , 0 ,width*height*sizeof(unsigned char));

    dim3 N_THREADS_PER_BLOCK(32, 32);
    dim3 N_BLOCKS((width + N_THREADS_PER_BLOCK.x - 1) / N_THREADS_PER_BLOCK.x, 
                  (height + N_THREADS_PER_BLOCK.y - 1) / N_THREADS_PER_BLOCK.y);
    
    rgb2gray_kernel<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(red_d, green_d, blue_d, gray_d, width, height);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
    }
    
    
    // Copy result back to host
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Save the grayscale image
    stbi_write_jpg("dog_grayscale.jpg", width, height, 1, gray, 90);
    printf("Grayscale image saved as dog_grayscale.jpg\n");
    
    // Free device memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    
    // Free host memory for gray image
    free(gray);
}

int main() {
    // File path
    const char* image_path = "./dog.jpg";
    
    // Image properties
    int width, height, channels;
    
    // Load the image using stb_image
    printf("Loading image: %s\n", image_path);
    unsigned char* img_data = stbi_load(image_path, &width, &height, &channels, 0);
    
    // Check if image loaded successfully
    if (img_data == NULL) {
        printf("Error loading image %s\n", image_path);
        printf("Make sure the file exists and the path is correct\n");
        return 1;
    }
    
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
    
    // Allocate memory for RGB channels
    unsigned char *red = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *green = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *blue = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    
    if (red == NULL || green == NULL || blue == NULL) {
        printf("Memory allocation failed\n");
        stbi_image_free(img_data);
        return 1;
    }
    
    // Extract RGB channels
    for (int i = 0; i < width * height; i++) {
        red[i] = img_data[i * channels + 0];
        green[i] = img_data[i * channels + 1];
        blue[i] = img_data[i * channels + 2];
    }
    
    printf("Channels extracted successfully\n");
    
    rgb2gray_gpu(red, green, blue, width, height);
    
    // Clean up
    free(red);
    free(green);
    free(blue);
    stbi_image_free(img_data);
    
    return 0;
}