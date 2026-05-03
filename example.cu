#include <cstdio>
#include "mallat.hpp"

#define SIZE 8
#define H_SIZE 2
#define MAX_LEVEL 3

float s[SIZE] = {4,6,10,12,8,6,5,5};
float h[H_SIZE] = {0.7071067811865475, 0.7071067811865475};

void show_array(float* arr){
    for(int i = 0; i < SIZE; i++){
        std::printf("%f ", arr[i]);
    }
    std::printf("\n");
}

int main(){
    
    std::printf("-=-=-Mallat's Algorithm Example-=-=-\n"); 
    std::printf("Signal: ");
    show_array(s);  
    std::printf("H filter: ");
    std::printf("%f %f\n", h[0], h[1]);
    std::printf("-----------------------\n");
    

    float* m;
    for(int i = 0; i < MAX_LEVEL; i++){
        int level = i+1;

        std::printf("Calculating Level: %d...\n",level);

        m = dtwt_level_n(level, H_SIZE, h, SIZE, s);
        show_array(m);
        if(level < MAX_LEVEL){
            cudaFreeHost(m);
        }
    }

    std::printf("-----------------------\n");

    float* mi;
    for(int i = MAX_LEVEL; i > 0; i--){
        int level = i-1;
        std::printf("Calculating inverse DTWT: %d...\n", level);
        mi = inverse_dtwt_level_n(level, H_SIZE, h, SIZE, m);
        show_array(mi);
        cudaFreeHost(mi);
    }

    cudaFreeHost(m);




    return 0;
}

