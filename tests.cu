#include <cassert>

#include "mallat.hpp"

namespace TestGenerateG{
    
    void TestGenerateG(){
        float h[2] = {0.7071067811865475, 0.7071067811865475};
        float* h_gpu = NULL;
        float* g_gpu = NULL;

        cudaError_t err1 = cudaMalloc((void**)&h_gpu, 2*sizeof(float));
        assert(err1 == cudaSuccess);

        cudaError_t err2 = cudaMalloc((void**)&g_gpu, 2*sizeof(float));
        assert(err2 == cudaSuccess);

        cudaError_t err3 = cudaMemcpy(h_gpu, h, 2*sizeof(float), cudaMemcpyHostToDevice);
        assert(err3 == cudaSuccess);
        
        generate_g_filter<<<1,2>>>(2, h_gpu, g_gpu);
        cudaDeviceSynchronize();
        
        float* g = (float*)malloc(2*sizeof(float));
        assert(g != NULL);

        cudaError_t err4 = cudaMemcpy(g, g_gpu, 2*sizeof(float), cudaMemcpyDeviceToHost);
        assert(err4 == cudaSuccess);

        assert(g[0] == h[0]);
        assert(g[1] == -h[1]);
    }


}


int main() {
   
    TestGenerateG::TestGenerateG(); 

    return 0;
}
