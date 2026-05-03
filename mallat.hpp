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
void inverse_dtwt(
    int signal_size,
    float* s, 
    float* h,
    float* g,
    float* m
){
    // we transpose DTWT
    int filter_index = blockIdx.x;

    int shift = (threadIdx.x/2)*2;
    int signal_index_result = (shift + blockIdx.x) % signal_size;
    
    int signal_index_calc = threadIdx.x;
    
    float filter_value = signal_index_calc % 2 == 0 ?
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

__global__ 
void get_correct_sequence(int half, float* m, float* nm){
    nm[blockIdx.x] = blockIdx.x % 2 == 0 ? m[(int)(blockIdx.x/2)] : m[(int)(blockIdx.x/2) + half];
}

__global__ 
void concat_to_previous(float* d, float* m){
    d[blockIdx.x] = m[blockIdx.x];
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
    assert(signal_size % 2 == 0); // NO ODD SIZED MATRICES
    assert(n > 0);

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

__host__ 
float* inverse_dtwt_level_n(int n, int filter_size, float* h, int dtwt_size, float* d){
    assert(dtwt_size % 2 == 0); // NO ODD SIZED MATRICES
    assert(n >= 0);

    float* h_gpu = copy_vector_to_gpu(filter_size, h); 
    float* g_gpu = init_gpu_array(filter_size);

    dim3 blocks_filter(filter_size < MAX_THREADS_PER_BLOCK ? 1 : std::ceil(filter_size/MAX_THREADS_PER_BLOCK));
    dim3 threads_filter(filter_size < MAX_THREADS_PER_BLOCK ? filter_size : MAX_THREADS_PER_BLOCK);
    generate_g_filter<<<blocks_filter, threads_filter>>>(filter_size, h_gpu, g_gpu);
    cudaDeviceSynchronize();

    float* d_gpu = copy_vector_to_gpu(dtwt_size, d);
    int current_level = std::log2(dtwt_size); 


    for(size_t i = current_level; i > n; i--){
        if(i <= 0) break;

        int level_signal_size = dtwt_size/std::pow(2,i);
        int complete_signal = 2*level_signal_size;

        float* m_gpu = init_gpu_array(complete_signal);
        float* seq = init_gpu_array(complete_signal);
        
        dim3 seq_blocks(complete_signal);
        dim3 seq_threads(1);
        get_correct_sequence<<<seq_blocks, seq_threads>>>(level_signal_size, d_gpu, seq);
        cudaDeviceSynchronize();

        inverse_dtwt<<<filter_size,complete_signal>>>(complete_signal, seq, h_gpu, g_gpu, m_gpu);
        cudaDeviceSynchronize();
        
        concat_to_previous<<<complete_signal, 1>>>(d_gpu, m_gpu);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaFree(m_gpu));
        CUDA_CHECK(cudaFree(seq));
    }

    CUDA_CHECK(cudaFree(h_gpu));
    CUDA_CHECK(cudaFree(g_gpu));

    float* m = (float*)malloc(dtwt_size*sizeof(float));
    if(m == NULL){
        std::printf("Failed on allocate memory for original signal\n");
        std::abort();
    }
    CUDA_CHECK(cudaMemcpy(m, d_gpu, dtwt_size*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_gpu));

    return m;
}


#ifdef DEBUG
}
#endif
