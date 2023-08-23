//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!


__global__ void radix_4_fft(float* x_r_d, float* x_i_d, unsigned int N, int M, unsigned int offset) {

	int i = tx + blockDim.x * (bx + gridDim.x * gridDim.y * bz + gridDim.x * by);

	float theta  = -2 * PI * (i % M) / (4 * M);
	int index = (i % M) + (i / M) * (4 * M);
	
	float x_1_r = x_r_d[index + offset];    
	float x_2_r = x_r_d[index + M + offset];
	float x_3_r = x_r_d[index + 2 * M + offset];
	float x_4_r = x_r_d[index + 3 * M + offset];
	
	float x_1_i = x_i_d[index + offset];
	float x_2_i = x_i_d[index + M + offset];
	float x_3_i = x_i_d[index + 2 * M + offset];
	float x_4_i = x_i_d[index + 3 * M + offset];	

	float y_1_r = x_2_r * cos(theta) - x_2_i * sin(theta);
	float y_2_r = x_3_r * cos(2 * theta) - x_3_i * sin(2 * theta);
	float y_3_r = x_4_r * cos(3 * theta) - x_4_i * sin(3 * theta);
	
	float y_1_i = x_2_r * sin(theta) + x_2_i * cos(theta);
	float y_2_i = x_3_r * sin(2 * theta) + x_3_i * cos(2 * theta);
	float y_3_i = x_4_r * sin(3 * theta) + x_4_i * cos(3 * theta);

	x_r_d[index + offset] = x_1_r + y_1_r + y_2_r + y_3_r;
	x_r_d[index + M + offset] = x_1_r + y_1_i - y_2_r - y_3_i;
	x_r_d[index + 2 * M + offset] = x_1_r - y_1_r + y_2_r - y_3_r;
	x_r_d[index + 3 * M + offset] = x_1_r - y_1_i - y_2_r + y_3_i;
	
	x_i_d[index + offset] = x_1_i + y_1_i + y_2_i + y_3_i;
	x_i_d[index + M + offset] = x_1_i - y_1_r - y_2_i + y_3_r;
	x_i_d[index + 2 * M + offset] = x_1_i - y_1_i + y_2_i - y_3_i;
	x_i_d[index + 3 * M + offset] = x_1_i + y_1_r - y_2_i - y_3_r;
}


__global__ void radix_2_fft(float* x_r_d, float* x_i_d, unsigned int N, int M) {
    int i = tx + blockDim.x * (bx + gridDim.x * gridDim.y * bz + gridDim.x * by);

    float theta = -2 * PI * ((i * (N / (2 * M))) - (i / M) * (N / 2)) / N;
    float W_i = sin(theta), W_r = cos(theta);

    float x_1_r = x_r_d[i + M * (i / M)];
    float x_2_r = x_r_d[i + M * (i / M + 1)];

    float x_1_i = x_i_d[i + M * (i / M)];
    float x_2_i = x_i_d[i + M * (i / M + 1)];

    x_r_d[i + M * (i / M)] = x_1_r + W_r * x_2_r - W_i * x_2_i;
	x_r_d[i + M * (i / M + 1)] = x_1_r + W_i * x_2_i - W_r * x_2_r;
	
    x_i_d[i + M * (i / M)] = x_1_i + W_r * x_2_i + W_i * x_2_r;
    x_i_d[i + M * (i / M + 1)] = x_1_i - W_i * x_2_r - W_r * x_2_i;
}



__global__ void sort_radix_4(float* x, float* y, unsigned int M, int offset) {
    unsigned int tid = tx + blockDim.x * (gridDim.x * gridDim.y * bz + gridDim.x * by + bx);
	unsigned int j = tid;
	
    // Perform radix-4 bit reordering
	j = ((j & 0b00110011001100110011001100110011) << 2) | ((j & 0b11001100110011001100110011001100) >> 2);
    j = ((j & 0b00001111000011110000111100001111) << 4) | ((j & 0b11110000111100001111000011110000) >> 4);
    j = ((j & 0b00000000111111110000000011111111) << 8) | ((j & 0b11111111000000001111111100000000) >> 8);	
    j = (j << 16) | (j >> 16);
    j >>= 32 - M;

	x[j] = y[tid + offset];
}


__global__ void transfer_array(float* x, float* y, int shift) {
	int i = tx + (bx + gridDim.x * gridDim.y * bz + gridDim.x * by) * blockDim.x;
	x[i + shift] = y[i];
}


void sort_evens(float* x_r_d, float* x_i_d, unsigned int N, unsigned int M, int offset) {
    float* y;
    cudaMalloc((void**) &y, N * sizeof(float));
	
	int gridDim1 = N / (1024 * 256);
	int gridDim2 = 32;
	int gridDim3 = 32;
	dim3 dimGrid(gridDim1, gridDim2, gridDim3);
	dim3 dimBlock(256, 1, 1);

    sort_radix_4 <<<dimGrid, dimBlock>>>(y, x_r_d, M, offset);
    transfer_array <<<dimGrid, dimBlock>>>(x_r_d, y, offset);

    sort_radix_4 <<<dimGrid, dimBlock>>>(y, x_i_d, M, offset);
    transfer_array <<<dimGrid, dimBlock>>>(x_i_d, y, offset);

    cudaFree(y);
}


__global__ void transfer_2_arrays(float* x, float* y, float* temp_x, float* temp_y) {
	int i = tx + (bx + gridDim.x * gridDim.y * bz + gridDim.x * by) * blockDim.x;
	x[i] = temp_x[i];
	y[i] = temp_y[i];
}


__global__ void transpose(float* x, float* y, float* temp_x, float* temp_y, unsigned int shift) {
    int i = tx + (bx + gridDim.x * gridDim.y * bz + gridDim.x * by) * blockDim.x;
	int index = i / 2 + (i % 2) * (shift / 2);
	x[index] = temp_x[i];
	y[index] = temp_y[i];
}

void sort_odds(float* x_r_d, float* x_i_d, unsigned int N, unsigned int M)
{
    float* y_r;
    float* y_i;

    cudaMalloc((void**) &y_r, N * sizeof(float));
    cudaMalloc((void**) &y_i, N * sizeof(float));

    dim3 dimGrid(N/1024, 1, 1);
	dim3 dimBlock(1024, 1, 1);
    transpose <<< dimGrid, dimBlock >>>(y_r, y_i, x_r_d, x_i_d, N);
    transfer_2_arrays <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, y_r, y_i);

    cudaFree(y_r);
    cudaFree(y_i);

    sort_evens(x_r_d, x_i_d, N / 2, M - 1, 0);
    sort_evens(x_r_d, x_i_d, N / 2, M - 1, N / 2);

}


void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M) {
	// In this function, both inputs and outputs are on GPU.
	// No need for cudaMalloc, cudaMemcpy or cudaFree.
	// This function does not run on GPU. 
	// You need to define another function and call it here for GPU execution.
	
	int blockDim1 = 128;
	int blockDim2 = 1;
	int blockDim3 = 1;
	dim3 dimBlock(blockDim1, blockDim2, blockDim3);

		
	if (M % 2 == 0) {
		sort_evens(x_r_d, x_i_d, N, M, 0);
		
		int gridDim1 = N / (256 * 16);
		int gridDim2 = 8;
		int gridDim3 = 1;
		dim3 dimGrid(gridDim1, gridDim2, gridDim3);
		
		for (int i = 1; i < N; i *= 4) {
			radix_4_fft <<< dimGrid, dimBlock >>> (x_r_d, x_i_d, N, i, 0);
		}
	} else {
		sort_odds(x_r_d, x_i_d, N, M);
	
		int gridDim1 = N / (256 * 32);
		int gridDim2 = 8;
		int gridDim3 = 1;
		dim3 dimGrid(gridDim1, gridDim2, gridDim3);
		
		for (int i = 1; i < N / 2; i *= 4) {
			radix_4_fft <<< dimGrid, dimBlock >>> (x_r_d, x_i_d, N, i, 0);
			radix_4_fft <<< dimGrid, dimBlock >>> (x_r_d, x_i_d, N, i, N/2);
		}
		
		int gridDim12 = N / (1024 * 512);
		int gridDim22 = 32;
		int gridDim32 = 32;
		dim3 dimGrid2(gridDim12, gridDim22, gridDim32);
		
		int blockDim1 = 256;
		int blockDim2 = 1;
		int blockDim3 = 1;
		dim3 dimBlock2(blockDim1, blockDim2, blockDim3);
		
		radix_2_fft <<< dimGrid2, dimBlock2 >>> (x_r_d, x_i_d, N, N / 2);
	}
}
