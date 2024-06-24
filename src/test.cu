#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Define perspective projection parameters
const float fov = 90.0f;    // Field of view
const float aspect = 1280.0f / 720.0f; // Aspect ratio
const float nearClip = 0.1f; // Near clipping plane
const float farClip = 100.0f; // Far clipping plane

// Function to compute perspective projection matrix
void setPerspective(float* projectionMatrix) {
    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);
    projectionMatrix[0] = f / aspect;
    projectionMatrix[5] = f;
    projectionMatrix[10] = (farClip + nearClip) / (nearClip - farClip);
    projectionMatrix[11] = -1.0f;
    projectionMatrix[14] = (2.0f * farClip * nearClip) / (nearClip - farClip);
}

void print_matrix(const string& name, const float *matrix, int rows, int cols) {
    cout << name << ": " << endl;
    cout << fixed << setprecision(2);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[j * rows + i] << "\t";
        }
        cout << endl;
    }
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define number of triangles
    const int NUM_TRIANGLES = 1;

    // Define the vertex data for one triangle (example)
    float vertices[4 * NUM_TRIANGLES * 3] = {  // 4 x NUM_TRIANGLES * 3
        -0.5f, -0.5f, 0.0f, 0.0f,
         0.5f, -0.5f, 0.0f, 0.0f,
         0.0f,  0.5f, 0.0f, 0.0f
    };
    print_matrix("Original vertices", vertices, 4, NUM_TRIANGLES * 3);
    float result[4 * NUM_TRIANGLES * 3]{0};

    // Allocate memory on GPU
    float* d_vertices;
    float* d_projectionMatrix;
    float* d_result;
    cudaMalloc((void**)&d_vertices, sizeof(vertices));
    cudaMalloc((void**)&d_projectionMatrix, sizeof(float) * 16);
    cudaMalloc((void**)&d_result, sizeof(result));

    // Copy data to GPU
    cudaMemcpy(d_vertices, vertices, sizeof(vertices), cudaMemcpyHostToDevice);

    // Set up projection matrix on GPU
    float projectionMatrix[16]{0}; // 4x4
    setPerspective(projectionMatrix);
    cudaMemcpy(d_projectionMatrix, projectionMatrix, sizeof(float) * 16, cudaMemcpyHostToDevice);

    // Compute perspective projection using CUBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, NUM_TRIANGLES * 3, 4, &alpha, d_projectionMatrix, 4, d_vertices, 4, &beta, d_result, 4);

    // Copy results back to host
    cudaMemcpy(result, d_result, sizeof(result), cudaMemcpyDeviceToHost);

    print_matrix("Resulting vertices", result, 4, NUM_TRIANGLES * 3);

    // Clean up
    cudaFree(d_vertices);
    cudaFree(d_projectionMatrix);
    cudaFree(d_result);
    cublasDestroy(handle);

    return 0;
}
