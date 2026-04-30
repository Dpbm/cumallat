#include <iostream>
#include <cuda_runtime_api.h>

extern "C" {


__device__ float s[8] = {4,6,10,12,8,6,5,5};
__device__ int filter_size = 2;
__device__ float h[2] = {0.7071067811865475, 0.7071067811865475};
__device__ float g[2] = {0.7071067811865475, -0.7071067811865475};
__device__ float m1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

__global__ void calculate(){
    int start_index = (blockIdx.x)/2;
    int end_index = (start_index + filter_size - 1) % blockDim.x; // handles the wrap around

    int filter_index = 0;
    if(start_index <= end_index && threadIdx.x >= start_index && threadIdx.x <= end_index){
        filter_index = threadIdx.x-start_index;
    }
    else if(start_index > end_index && (threadIdx.x >= start_index || threadIdx.x <= end_index)){
        filter_index = threadIdx.x >= start_index ? 
                        threadIdx.x-start_index :
                        threadIdx.x + (filter_size - (start_index - blockDim.x));
    }
    else{
        return;
    }
    
    float filter_value = blockIdx.x % 2 == 0 ?
        h[filter_index]: 
        g[filter_index];

    atomicAdd(&m1[blockIdx.x], filter_value * s[blockIdx.x]);
}

}

int main(){
    calculate<<<8,8>>>();
    cudaDeviceSynchronize();

    float* local_m = (float*)malloc(8*sizeof(float));
    cudaMemcpyFromSymbol(local_m, m1, 8*sizeof(float));
    
    for(size_t i = 0; i < 8; i++){
        std::cout << local_m[i] << ", ";
    }
    std::cout << std::endl;


    return 0;
}

