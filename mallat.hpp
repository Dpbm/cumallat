#pragma once

#include <cstdio>
#include <cmath>
#include <cuda_runtime_api.h>

#define DEBUG

#ifdef DEBUG
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024

#define CUDA_CHECK(X) \
    do{ \
        cudaError_t err = X; \
        if(err != cudaSuccess){ \
            std::printf("Failed on Evaluate cuda function:\nError Name: %s\nError String: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
            std::abort(); \
        } \
    }while(0);

__host__
float* copy_vector_to_gpu(int size, float* v){
    float* v_gpu = NULL;
    CUDA_CHECK(cudaMalloc((void**)&v_gpu, size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(v_gpu, v, size*sizeof(float), cudaMemcpyHostToDevice));
    return v_gpu;
}

__global__
void generate_g_filter(int filter_size, float* h, float* g){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int inverse_index = filter_size - 1 - index;
    g[inverse_index] = h[index] * (inverse_index % 2 == 1 ? -1 : 1);
}

__global__
void dtwt(
    float* s, 
    int filter_size,
    float* h,
    float* g,
    float* m
){
    int filter_index = threadIdx.x;

    int signal_index_result = blockIdx.x;
    int signal_index_calc = ((blockIdx.x/2)*2) + threadIdx.x;
    
    float filter_value = signal_index_result % 2 == 0 ?
        h[filter_index]: 
        g[filter_index];
        
    atomicAdd(&m[signal_index_result], filter_value * s[signal_index_calc]);
}

__host__ 
float* dtwt_level_n(int n, int filter_size, float* h, int signal_size, float* s){
    float* h_gpu = copy_vector_to_gpu(filter_size, h); 

    float* g_gpu = NULL;
    CUDA_CHECK(cudaMalloc((void**)&g_gpu, filter_size*sizeof(float)));

    dim3 blocks_filter(filter_size < MAX_THREADS_PER_BLOCK ? 1 : std::ceil(filter_size/MAX_THREADS_PER_BLOCK));
    dim3 threads_filter(filter_size < MAX_THREADS_PER_BLOCK ? filter_size : MAX_THREADS_PER_BLOCK);
    generate_g_filter<<<blocks_filter, threads_filter>>>(filter_size, h_gpu, g_gpu);
    cudaDeviceSynchronize();
    
    float* s_gpu = copy_vector_to_gpu(signal_size, s);

    float* m_gpu = NULL;
    CUDA_CHECK(cudaMalloc((void**)&m_gpu, signal_size*sizeof(float)));

    dtwt<<<signal_size, filter_size>>>(s_gpu, filter_size, h_gpu, g_gpu, m_gpu);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaFree(h_gpu));
    CUDA_CHECK(cudaFree(g_gpu));
    CUDA_CHECK(cudaFree(s_gpu));

    float* m = (float*)malloc(signal_size*sizeof(float));
    if(m == NULL){
        std::printf("Failed on allocate memory for processed signal\n");
        std::abort();
    }
    CUDA_CHECK(cudaMemcpy(m, m_gpu, signal_size*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(m_gpu));

    return m;
}



#ifdef DEBUG
}
#endif
