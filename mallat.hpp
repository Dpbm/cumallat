#pragma once

#include <cstdio>
#include <cassert>
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

__device__ __forceinline__
int get_index(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__host__
float* copy_vector_to_gpu(int size, float* v){
    float* v_gpu = NULL;
    CUDA_CHECK(cudaMalloc((void**)&v_gpu, size*sizeof(float)));
    CUDA_CHECK(cudaMemset(v_gpu, 0, size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(v_gpu, v, size*sizeof(float), cudaMemcpyHostToDevice));
    return v_gpu;
}

__global__
void generate_g_filter(int filter_size, float* h, float* g){
    int index = get_index();
    if(index >= filter_size) return;

    int inverse_index = filter_size - 1 - index;
    g[inverse_index] = h[index] * (inverse_index % 2 == 1 ? -1 : 1);
}

__global__
void dtwt(
    int signal_size,
    float* s, 
    float* h,
    float* g,
    float* m
){
    int filter_index = threadIdx.x;
    int signal_index_result = blockIdx.x;

    int shift = (blockIdx.x/2)*2;
    int signal_index_calc = (shift + threadIdx.x) % signal_size;
    
    float filter_value = signal_index_result % 2 == 0 ?
        h[filter_index]: 
        g[filter_index];
        
    atomicAdd(&m[signal_index_result], filter_value * s[signal_index_calc]);
}

__global__ 
void organize_m(int half, float* m, float* nm){
    int index = get_index();
    if(index >= half*2) return;

    int nm_index = (index/2) + (index % 2 == 1 ? half : 0);
    nm[nm_index] = m[index];
}

__host__ 
float* init_gpu_array(int size){
    float* v = NULL;
    CUDA_CHECK(cudaMalloc((void**)&v, size*sizeof(float)));
    CUDA_CHECK(cudaMemset(v, 0, size*sizeof(float)));
    return v;
}

__host__
void reset_array(float* arr, int size){
    CUDA_CHECK(cudaMemset(arr, 0, size*sizeof(float)));
}

__host__ 
float* dtwt_level_n(int n, int filter_size, float* h, int signal_size, float* s){
    assert(signal_size % 2 == 0); // TODO: REMOVE LATER

    float* h_gpu = copy_vector_to_gpu(filter_size, h); 
    float* g_gpu = init_gpu_array(filter_size);

    dim3 blocks_filter(filter_size < MAX_THREADS_PER_BLOCK ? 1 : std::ceil(filter_size/MAX_THREADS_PER_BLOCK));
    dim3 threads_filter(filter_size < MAX_THREADS_PER_BLOCK ? filter_size : MAX_THREADS_PER_BLOCK);
    generate_g_filter<<<blocks_filter, threads_filter>>>(filter_size, h_gpu, g_gpu);
    cudaDeviceSynchronize();
    
    float* s_gpu = copy_vector_to_gpu(signal_size, s);
    float* m_gpu = init_gpu_array(signal_size);
    float* m_organized_gpu = init_gpu_array(signal_size);

    for(size_t i = 0; i < n; i++){

        // TODO: verify if we should stop
        int level_signal_size = i == 0 ? signal_size : signal_size/std::pow(2,i);

        if(level_signal_size <= 1) break;
        
        if(i == 0){
            dtwt<<<level_signal_size, filter_size>>>(level_signal_size, s_gpu, h_gpu, g_gpu, m_gpu);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaFree(s_gpu));
        }else{
            reset_array(m_gpu, signal_size);

            dtwt<<<level_signal_size, filter_size>>>(level_signal_size, m_organized_gpu, h_gpu, g_gpu, m_gpu);
            cudaDeviceSynchronize();
        }
        
        dim3 blocks_organize(level_signal_size < MAX_THREADS_PER_BLOCK ? 1 : std::ceil(level_signal_size/MAX_THREADS_PER_BLOCK));
        dim3 threads_organize(level_signal_size < MAX_THREADS_PER_BLOCK ? level_signal_size : MAX_THREADS_PER_BLOCK);
        organize_m<<<blocks_organize, threads_organize>>>(level_signal_size/2, m_gpu, m_organized_gpu);
        cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaFree(h_gpu));
    CUDA_CHECK(cudaFree(g_gpu));
    CUDA_CHECK(cudaFree(m_gpu));

    float* m = (float*)malloc(signal_size*sizeof(float));
    if(m == NULL){
        std::printf("Failed on allocate memory for processed signal\n");
        std::abort();
    }
    CUDA_CHECK(cudaMemcpy(m, m_organized_gpu, signal_size*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(m_organized_gpu));

    return m;
}


/* TODOS
 * 1. organize the resulting data (per level) [X]
 * 2. wraparound                              [X]
 * 3. filters that are bigger than signal
 * 4. odd sized signals
 */

#ifdef DEBUG
}
#endif
