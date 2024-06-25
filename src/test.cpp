#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include "test.hpp"

using namespace std;

void perspective_projection(float *result, float *vertices, int cols, float fov, float aspect, float nearClip, float farClip) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // rows of input is 3
    const int rows = 3;
    const int _rows = 4;

    // Set up projection matrix on GPU
    float projection_matrix[16]{0}; // 4x4
    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);
    projection_matrix[0] = f / aspect;
    projection_matrix[5] = f;
    projection_matrix[10] = (farClip + nearClip) / (nearClip - farClip);
    projection_matrix[11] = -1.0f;
    projection_matrix[14] = (2.0f * farClip * nearClip) / (nearClip - farClip);

    // print_matrix("Original vertices", vertices, rows, cols);

    // make new vertices array with 4 rows instead of 3
    float *_vertices = new float[_rows * cols]{1.0f};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            _vertices[j * _rows + i] = vertices[j * rows + i];
        }
    }
    for (int j = 0; j < cols; j++) {
        _vertices[j * _rows + 3] = 1.0f;
    }

    // print_matrix("Augmented vertices", _vertices, _rows, cols);

    // Allocate memory on GPU
    float* gpu_vertices;
    float* gpu_projection_matrix;
    float* gpu_result;
    cudaMalloc((void**)&gpu_vertices, _rows * cols * sizeof(float));
    cudaMalloc((void**)&gpu_projection_matrix, 16 * sizeof(float));
    cudaMalloc((void**)&gpu_result, _rows * cols * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(gpu_vertices, _vertices, _rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_projection_matrix, projection_matrix, 16 * sizeof(float), cudaMemcpyHostToDevice);

    // Compute perspective projection using CUBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, _rows, cols, _rows, &alpha, gpu_projection_matrix, _rows, gpu_vertices, _rows, &beta, gpu_result, _rows);

    // make new vertices array with 4 rows instead of 3
    float *_result = new float[_rows * cols]{0};

    // Copy results back to host
    cudaMemcpy(_result, gpu_result, _rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = _result[j * _rows + i]/_result[j * _rows + 3];
        }
    }

    // print_matrix("Resulting augmented vertices", _result, _rows, cols);

    // print_matrix("Resulting vertices", result, rows, cols);

    // Clean up
    cudaFree(gpu_vertices);
    cudaFree(gpu_projection_matrix);
    cudaFree(gpu_result);
    delete[] _vertices;
    delete[] _result;
    cublasDestroy(handle);
}