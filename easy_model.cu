#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

//CUDA简易模型

const int INPUT_SIZE = 2;
const int HIDDEN_SIZE = 3;
const int OUTPUT_SIZE = 1;
const int BATCH_SIZE = 1;
const float LEARNING_RATE = 0.1f;
const int ITERATIONS = 1000;

__global__ void sigmoid(float* in, float* out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void matrixMultiply(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void addBias(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

__global__ void derivativeSigmoid(float* in, float* out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float sigmoid_value = 1.0f / (1.0f + expf(-in[idx]));
        out[idx] = sigmoid_value * (1.0f - sigmoid_value);
    }
}

__global__ void updateWeights(float* weights, float* gradients, int size, float learning_rate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

int main() {
    // Initialize host memory
    float* h_input = new float[INPUT_SIZE * BATCH_SIZE];
    float* h_output = new float[OUTPUT_SIZE * BATCH_SIZE];

    // Initialize input and output
    h_input[0] = 0.0f;  // Input 1
    h_input[1] = 1.0f;  // Input 2
    h_output[0] = 1.0f; // Expected output

    // Allocate device memory
    float* d_input, * d_hidden_weights, * d_hidden_bias, * d_hidden_output;
    float* d_output_weights, * d_output_bias, * d_output, * d_target, * d_error;

    cudaMalloc((void**)&d_input, INPUT_SIZE * BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&d_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_hidden_bias, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_hidden_output, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output_bias, OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&d_target, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&d_error, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize weights and biases
    float h_hidden_weights[INPUT_SIZE * HIDDEN_SIZE] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    float h_hidden_bias[HIDDEN_SIZE] = { 0.1f, 0.2f, 0.3f };
    float h_output_weights[HIDDEN_SIZE * OUTPUT_SIZE] = { 0.7f, 0.8f, 0.9f };
    float h_output_bias[OUTPUT_SIZE] = { 0.4f };

    cudaMemcpy(d_hidden_weights, h_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_bias, h_hidden_bias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, h_output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_bias, h_output_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGridHidden((HIDDEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, BATCH_SIZE);
    dim3 blocksPerGridOutput((OUTPUT_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, BATCH_SIZE);

// Training loop
for (int iter = 0; iter < ITERATIONS; ++iter) {
    // Forward pass
    matrixMultiply <<<blocksPerGridHidden, threadsPerBlock >>> (d_input, d_hidden_weights, d_hidden_output, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
    addBias <<<blocksPerGridHidden, threadsPerBlock >>> (d_hidden_output, d_hidden_bias, HIDDEN_SIZE * BATCH_SIZE);
    sigmoid <<<blocksPerGridHidden, threadsPerBlock >>> (d_hidden_output, d_hidden_output, HIDDEN_SIZE * BATCH_SIZE);

    matrixMultiply <<<blocksPerGridOutput, threadsPerBlock >>> (d_hidden_output, d_output_weights, d_output, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    addBias <<<blocksPerGridOutput, threadsPerBlock >>> (d_output, d_output_bias, OUTPUT_SIZE * BATCH_SIZE);
    sigmoid <<<blocksPerGridOutput, threadsPerBlock >>> (d_output, d_output, OUTPUT_SIZE * BATCH_SIZE);

    // Compute error
    cudaMemcpy(d_target, h_output, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_error, d_output, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_error, d_target, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    derivativeSigmoid <<<blocksPerGridOutput, threadsPerBlock >>> (d_output, d_error, OUTPUT_SIZE * BATCH_SIZE);
    matrixMultiply <<<blocksPerGridOutput, threadsPerBlock >>> (d_error, d_output_weights, d_hidden_output, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);

    // Backward pass
    derivativeSigmoid <<<blocksPerGridHidden, threadsPerBlock >>> (d_hidden_output, d_hidden_output, HIDDEN_SIZE * BATCH_SIZE);
    matrixMultiply <<<blocksPerGridHidden, threadsPerBlock >>> (d_hidden_output, d_error, d_hidden_weights, HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE);

    // Update weights
    updateWeights <<<(INPUT_SIZE * HIDDEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock >>> (d_hidden_weights, d_hidden_output, INPUT_SIZE * HIDDEN_SIZE, LEARNING_RATE);
    updateWeights <<<(HIDDEN_SIZE * OUTPUT_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock >>> (d_output_weights, d_output, HIDDEN_SIZE * OUTPUT_SIZE, LEARNING_RATE);
}

// Cleanup
delete[] h_input;
delete[] h_output;
cudaFree(d_input);
cudaFree(d_hidden_weights);
cudaFree(d_hidden_bias);
cudaFree(d_hidden_output);
cudaFree(d_output_weights);
cudaFree(d_output_bias);
cudaFree(d_output);
cudaFree(d_target);
cudaFree(d_error);

return 0;
}