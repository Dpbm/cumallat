#include <cassert>

#include "mallat.hpp"

extern "C" {

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

namespace Organization{
    void TestOrganizeM(){
        float m[4] = {0,1,2,3};
        float* m_gpu = NULL;
        float* m_new = NULL;

        cudaError_t err1 = cudaMalloc((void**)&m_gpu, 4*sizeof(float));
        assert(err1 == cudaSuccess);

        cudaError_t err2 = cudaMalloc((void**)&m_new, 4*sizeof(float));
        assert(err2 == cudaSuccess);

        cudaError_t err3 = cudaMemcpy(m_gpu, m, 4*sizeof(float), cudaMemcpyHostToDevice);
        assert(err3 == cudaSuccess);

        organize_m<<<2,2>>>(2, m_gpu, m_new);
        cudaDeviceSynchronize();
        
        float* m_new_cpu = (float*)malloc(4*sizeof(float));
        assert(m_new_cpu != NULL);

        cudaError_t err4 = cudaMemcpy(m_new_cpu, m_new, 4*sizeof(float), cudaMemcpyDeviceToHost);
        assert(err4 == cudaSuccess);

        assert(m_new_cpu[0] == m[0]);
        assert(m_new_cpu[1] == m[2]);
        assert(m_new_cpu[2] == m[1]);
        assert(m_new_cpu[3] == m[3]);
    }
}

namespace DTWT{
    
    void TestDTWTHaarFilters(){
        
        float sqrt_2 = 1.4142135623730951;
        float s[8] = {sqrt_2, sqrt_2,  sqrt_2, sqrt_2,sqrt_2, sqrt_2,  sqrt_2, sqrt_2};
        float h[2] = {0.7071067811865475, 0.7071067811865475};

        float* result;
        
        result = dtwt_level_n(1, 2, h, 8, s);
        assert(result[0]-2 <= 0.1);
        assert(result[1]-2 <= 0.1);
        assert(result[2]-2 <= 0.1);
        assert(result[3]-2 <= 0.1);
        assert(result[4] == 0);
        assert(result[5] == 0);
        assert(result[6] == 0);
        assert(result[7] == 0);
        
        
        result = dtwt_level_n(2, 2, h, 8, s);
        assert(result[0]-2.82842712474619 <= 0.1);
        assert(result[1]-2.82842712474619 <= 0.1);
        assert(result[2] == 0);
        assert(result[3] == 0);
        assert(result[4] == 0);
        assert(result[5] == 0);
        assert(result[6] == 0);
        assert(result[7] == 0);
        
        result = dtwt_level_n(3, 2, h, 8, s);
        assert(result[0]-4 <= 0.1);
        assert(result[1] == 0);
        assert(result[2] == 0);
        assert(result[3] == 0);
        assert(result[4] == 0);
        assert(result[5] == 0);
        assert(result[6] == 0);
        assert(result[7] == 0);
        
        result = dtwt_level_n(4, 2, h, 8, s);
        assert(result[0]-4 <= 0.1);
        assert(result[1] == 0);
        assert(result[2] == 0);
        assert(result[3] == 0);
        assert(result[4] == 0);
        assert(result[5] == 0);
        assert(result[6] == 0);
        assert(result[7] == 0);
    }

}

}

int main() {
   
    TestGenerateG::TestGenerateG(); 
    Organization::TestOrganizeM();
    DTWT::TestDTWTHaarFilters();

    return 0;
}
