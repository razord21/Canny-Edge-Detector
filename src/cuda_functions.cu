#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_functions.h"

#define VERBOSE 1
#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0
#define BOOSTBLURFACTOR 90.0
#define MAX_KERNEL_SIZE 32
#define BLOCK_SIZE 16

__constant__ float c_gaussian_kernel[MAX_KERNEL_SIZE];

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while (0)

__global__ void gaussian_smooth_kernel(unsigned char *image, short int *smoothedim, int rows, int cols, float *kernel, int windowsize) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int center = windowsize / 2;
    float dot = 0.0, sum = 0.0;
    if (r < rows && c < cols) {
        for (int cc = -center; cc <= center; cc++) {
            if ((c + cc) >= 0 && (c + cc) < cols) {
                dot += image[r * cols + (c + cc)] * kernel[center + cc];
                sum += kernel[center + cc];
            }
        }
        float temp = dot / sum;
        dot = 0.0; sum = 0.0;
        for (int rr = -center; rr <= center; rr++) {
            if ((r + rr) >= 0 && (r + rr) < rows) {
                dot += temp * kernel[center + rr];
                sum += kernel[center + rr];
            }
        }
        smoothedim[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
    }
}

void make_gaussian_kernel(float sigma, float **kernel, int *windowsize) {
    int i, center; float x, fx, sum = 0.0;
    *windowsize = 1 + 2 * ceil(2.5 * sigma);
    center = (*windowsize) / 2;
    if (*windowsize > MAX_KERNEL_SIZE) { fprintf(stderr, "Error: Kernel size (%d) exceeds MAX_KERNEL_SIZE (%d).\n", *windowsize, MAX_KERNEL_SIZE); exit(1); }
    if (VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
    if ((*kernel = (float *)malloc((*windowsize) * sizeof(float))) == NULL) { fprintf(stderr, "Error allocating the Gaussian kernel array.\n"); exit(1); }
    for (i = 0; i < (*windowsize); i++) { x = (float)(i - center); fx = pow(2.71828, -0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(6.2831853)); (*kernel)[i] = fx; sum += fx; }
    for (i = 0; i < (*windowsize); i++) (*kernel)[i] /= sum;
    if (VERBOSE) { printf("The filter coefficients are:\n"); for (i = 0; i < (*windowsize); i++) printf("kernel[%d] = %f\n", i, (*kernel)[i]); }
}

void cuda_gaussian_smooth(unsigned char* image, int rows, int cols, float sigma, short int** smoothedim) {
    int windowsize; float* kernel; make_gaussian_kernel(sigma, &kernel, &windowsize);
    unsigned char* d_image; short int* d_smoothedim; float* d_kernel;
    CHECK_CUDA(cudaMalloc((void**)&d_image, rows * cols * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc((void**)&d_smoothedim, rows * cols * sizeof(short int)));
    CHECK_CUDA(cudaMalloc((void**)&d_kernel, windowsize * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_image, image, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel, windowsize * sizeof(float), cudaMemcpyHostToDevice));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gaussian_smooth_kernel<<<dimGrid, dimBlock>>>(d_image, d_smoothedim, rows, cols, d_kernel, windowsize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(*smoothedim, d_smoothedim, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost));
    printf("First 10 smoothed pixel values:\n"); for (int i = 0; i < 10; i++) { printf("%d ", (*smoothedim)[i]); } printf("\n");
    cudaFree(d_image); cudaFree(d_smoothedim); cudaFree(d_kernel); free(kernel);
}

__global__ void cuda_derivative_x(const short int *smoothedim, short int *delta_x, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c;
        if (c == 0) { delta_x[pos] = smoothedim[pos + 1] - smoothedim[pos]; }
        else if (c == cols - 1) { delta_x[pos] = smoothedim[pos] - smoothedim[pos - 1]; }
        else { delta_x[pos] = smoothedim[pos + 1] - smoothedim[pos - 1]; }
    }
}

__global__ void cuda_derivative_y(const short int *smoothedim, short int *delta_y, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c;
        if (r == 0) { delta_y[pos] = smoothedim[pos + cols] - smoothedim[pos]; }
        else if (r == rows - 1) { delta_y[pos] = smoothedim[pos] - smoothedim[pos - cols]; }
        else { delta_y[pos] = smoothedim[pos + cols] - smoothedim[pos - cols]; }
    }
}

void cuda_derivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y)
{
    *delta_x = (short int *)malloc(rows * cols * sizeof(short int));
    *delta_y = (short int *)malloc(rows * cols * sizeof(short int));
    if (*delta_x == NULL || *delta_y == NULL) { fprintf(stderr, "Error allocating memory for derivatives.\n"); exit(1); }
    short int *d_smoothedim, *d_delta_x, *d_delta_y;
    cudaMalloc((void **)&d_smoothedim, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_delta_x, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_delta_y, rows * cols * sizeof(short int));
    cudaMemcpy(d_smoothedim, smoothedim, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    cuda_derivative_x<<<gridSize, blockSize>>>(d_smoothedim, d_delta_x, rows, cols);
    cuda_derivative_y<<<gridSize, blockSize>>>(d_smoothedim, d_delta_y, rows, cols);
    cudaMemcpy(*delta_x, d_delta_x, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*delta_y, d_delta_y, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaFree(d_smoothedim); cudaFree(d_delta_x); cudaFree(d_delta_y);
}

__global__ void cuda_radian_direction_kernel(const short int *delta_x, const short int *delta_y, float *dir_radians, int rows, int cols, int xdirtag, int ydirtag)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c; double dx = (double)delta_x[pos]; double dy = (double)delta_y[pos];
        if (xdirtag == 1) dx = -dx; if (ydirtag == -1) dy = -dy; dir_radians[pos] = (float)atan2(dy, dx);
    }
}

void cuda_radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag)
{
    *dir_radians = (float *)malloc(rows * cols * sizeof(float)); if (*dir_radians == NULL) { fprintf(stderr, "Error allocating memory for gradient direction.\n"); exit(1); }
    short int *d_delta_x, *d_delta_y; float *d_dir_radians;
    cudaMalloc((void **)&d_delta_x, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_delta_y, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_dir_radians, rows * cols * sizeof(float));
    cudaMemcpy(d_delta_x, delta_x, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_y, delta_y, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16); dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    cuda_radian_direction_kernel<<<gridSize, blockSize>>>(d_delta_x, d_delta_y, d_dir_radians, rows, cols, xdirtag, ydirtag);
    cudaMemcpy(*dir_radians, d_dir_radians, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_delta_x); cudaFree(d_delta_y); cudaFree(d_dir_radians);
}

__global__ void cuda_magnitude_x_y_kernel(const short int *delta_x, const short int *delta_y, short int *magnitude, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c; int sq1 = (int)delta_x[pos] * (int)delta_x[pos]; int sq2 = (int)delta_y[pos] * (int)delta_y[pos];
        magnitude[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
    }
}

void cuda_magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude)
{
    *magnitude = (short int *)malloc(rows * cols * sizeof(short int)); if (*magnitude == NULL) { fprintf(stderr, "Error allocating memory for magnitude image.\n"); exit(1); }
    short int *d_delta_x, *d_delta_y, *d_magnitude;
    cudaMalloc((void **)&d_delta_x, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_delta_y, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_magnitude, rows * cols * sizeof(short int));
    cudaMemcpy(d_delta_x, delta_x, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_y, delta_y, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16); dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    cuda_magnitude_x_y_kernel<<<gridSize, blockSize>>>(d_delta_x, d_delta_y, d_magnitude, rows, cols);
    cudaMemcpy(*magnitude, d_magnitude, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaFree(d_delta_x); cudaFree(d_delta_y); cudaFree(d_magnitude);
}

#define MAX_QUEUE_SIZE 1000000
__device__ int queue_arr[MAX_QUEUE_SIZE];
__device__ int queue_start = 0;
__device__ int queue_end = 0;

__device__ void enqueue(int pos)
{ int end = atomicAdd(&queue_end, 1); if (end < MAX_QUEUE_SIZE) { queue_arr[end] = pos; } }
__device__ int dequeue_pos()
{ int start = atomicAdd(&queue_start, 1); if (start < queue_end) { return queue_arr[start]; } return -1; }

__global__ void reset_queue()
{ queue_start = 0; queue_end = 0; }

__global__ void cuda_initialize_edges(const unsigned char *nms, unsigned char *edge, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c; edge[pos] = (nms[pos] == POSSIBLE_EDGE) ? POSSIBLE_EDGE : NOEDGE;
        if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) { edge[pos] = NOEDGE; }
    }
}

__global__ void cuda_compute_histogram(const short int *mag, const unsigned char *edge, int *hist, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c; if (edge[pos] == POSSIBLE_EDGE) { atomicAdd(&hist[mag[pos]], 1); }
    }
}

__global__ void cuda_apply_hysteresis_kernel(const short int *mag, unsigned char *edge, int lowthreshold, int highthreshold, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c;
        if (edge[pos] == POSSIBLE_EDGE)
        {
            if (mag[pos] >= highthreshold) { edge[pos] = EDGE; enqueue(pos); }
            else if (mag[pos] < lowthreshold) { edge[pos] = NOEDGE; }
        }
    }
}

__global__ void cuda_follow_edges(unsigned char *edgemap, short *edgemag, int lowval, int rows, int cols)
{
    int pos = dequeue_pos(); if (pos == -1) return; int r = pos / cols; int c = pos % cols;
    int x[8] = {1, 1, 0, -1, -1, -1, 0, 1}; int y[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int i = 0; i < 8; i++)
    {
        int nr = r + y[i]; int nc = c + x[i]; if (nr >= 0 && nr < rows && nc >= 0 && nc < cols)
        {
            int npos = nr * cols + nc;
            if (edgemap[npos] == POSSIBLE_EDGE && edgemag[npos] > lowval)
            { edgemap[npos] = EDGE; enqueue(npos); }
        }
    }
}

__global__ void cuda_final_cleanup(unsigned char *edge, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
    {
        int pos = r * cols + c; if (edge[pos] != EDGE) { edge[pos] = NOEDGE; }
    }
}

void cuda_apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge)
{
    short int *d_mag; unsigned char *d_nms, *d_edge; int *d_hist;
    cudaMalloc((void **)&d_mag, rows * cols * sizeof(short int));
    cudaMalloc((void **)&d_nms, rows * cols * sizeof(unsigned char));
    cudaMalloc((void **)&d_edge, rows * cols * sizeof(unsigned char));
    cudaMalloc((void **)&d_hist, 32768 * sizeof(int));
    cudaMemcpy(d_mag, mag, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nms, nms, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16); dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    cuda_initialize_edges<<<gridSize, blockSize>>>(d_nms, d_edge, rows, cols);
    cudaMemset(d_hist, 0, 32768 * sizeof(int));
    cuda_compute_histogram<<<gridSize, blockSize>>>(d_mag, d_edge, d_hist, rows, cols);
    int hist[32768]; cudaMemcpy(hist, d_hist, 32768 * sizeof(int), cudaMemcpyDeviceToHost);
    int numedges = 0, maximum_mag = 0; for (int r = 1; r < 32768; r++) { if (hist[r] != 0) maximum_mag = r; numedges += hist[r]; }
    int highcount = (int)(numedges * thigh + 0.5); int r = 1; numedges = hist[1];
    while ((r < (maximum_mag - 1)) && (numedges < highcount)) { r++; numedges += hist[r]; }
    int highthreshold = r; int lowthreshold = (int)(highthreshold * tlow + 0.5);
    cuda_apply_hysteresis_kernel<<<gridSize, blockSize>>>(d_mag, d_edge, lowthreshold, highthreshold, rows, cols);
    reset_queue<<<1, 1>>>();
    cuda_follow_edges<<<gridSize, blockSize>>>(d_edge, d_mag, lowthreshold, rows, cols);
    cuda_final_cleanup<<<gridSize, blockSize>>>(d_edge, rows, cols);
    cudaMemcpy(edge, d_edge, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_mag); cudaFree(d_nms); cudaFree(d_edge); cudaFree(d_hist);
}
