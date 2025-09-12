#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void cuda_gaussian_smooth(unsigned char* image, int rows, int cols, float sigma, short int** smoothedim);
void cuda_derivative_x_y(short int* smoothedim, int rows, int cols, short int** delta_x, short int** delta_y);
void cuda_radian_direction(short int* delta_x, short int* delta_y, int rows, int cols, float** dir_radians, int xdirtag, int ydirtag);
void cuda_magnitude_x_y(short int* delta_x, short int* delta_y, int rows, int cols, short int** magnitude);
void cuda_apply_hysteresis(short int* mag, unsigned char* nms, int rows, int cols, float tlow, float thigh, unsigned char* edge);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FUNCTIONS_H
