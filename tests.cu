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

    void TestWrapAroundFilters(){
        float s[8] = {2,1,4,3,5,2,1,0};
        float h[4] = {2,3,2,3};

        float* result;
        result = dtwt_level_n(1, 4, h, 8, s);
        assert(result[0] == 24);
        assert(result[1] == 33);
        assert(result[2] == 18);
        assert(result[3] == 9);
        assert(result[4] == 10);
        assert(result[5] == 17);
        assert(result[6] == 14);
        assert(result[7] == 7);
    
    }

    void TestFilterBiggerThanSignal(){
        float s1[2] = {2,1};
        float h1[4] = {2,3,2,3};

        float* result;
        result = dtwt_level_n(1, 4, h1, 2, s1);
        assert(result[0] == 14);
        assert(result[1] == 8);
        
        float s2[4] = {1,2,3,4};
        float h2[6] = {1,2,3,4,5,6};
        result = dtwt_level_n(1, 6, h2, 4, s2);
        assert(result[0] == 47);
        assert(result[1] == 61);
        assert(result[2] == -4);
        assert(result[3] == -2);
    }


    void TestInverse(){
        float d[4] = {27,-1,0,2};
        float h[2] = {1,2};

        float* result;

        result = inverse_dtwt_level_n(1, 2, h, 4, d);
        assert(result[0] == 25);
        assert(result[1] == 55);
        assert(result[2] == 0);
        assert(result[3] == 2);
        
        result = inverse_dtwt_level_n(0, 2, h, 4, d);
        assert(result[0] == 25);
        assert(result[1] == 50);
        assert(result[2] == 59);
        assert(result[3] == 108);

    }

    void TestInverseWrapAround(){
        float d1[8] = {4472,1144,76,188,0,4,8,1};
        float h1[4] = {1,2,3,4};
        float* result;
        result = inverse_dtwt_level_n(2, 4, h1, 8, d1);
        assert(result[0] == 24752);
        assert(result[1] == 22256);
        assert(result[2] == 76);
        assert(result[3] == 188);
        assert(result[4] == 0);
        assert(result[5] == 4);
        assert(result[6] == 8);
        assert(result[7] == 1);
        
        result = inverse_dtwt_level_n(1, 4, h1, 8, d1);
        assert(result[0] == 92200);
        assert(result[1] == 138112);
        assert(result[2] == 97416);
        assert(result[3] == 142880);
        assert(result[4] == 0);
        assert(result[5] == 4);
        assert(result[6] == 8);
        assert(result[7] == 1);
        
        result = inverse_dtwt_level_n(0, 4, h1, 8, d1);
        assert(result[0] == 520842);
        assert(result[1] == 755919);
        assert(result[2] == 414728);
        assert(result[3] == 645012);
        assert(result[4] == 511792);
        assert(result[5] == 747252);
        assert(result[6] == 435148);
        assert(result[7] == 675413);
    }
}

}

int main() {
    TestGenerateG::TestGenerateG(); 
    Organization::TestOrganizeM();
    DTWT::TestDTWTHaarFilters();
    DTWT::TestWrapAroundFilters();
    DTWT::TestFilterBiggerThanSignal();
    DTWT::TestInverse();
    DTWT::TestInverseWrapAround();

    return 0;
}
