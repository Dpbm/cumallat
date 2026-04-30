#include <iostream>
#include "mallat.hpp"

float s[8] = {4,6,10,12,8,6,5,5};
float h[2] = {0.7071067811865475, 0.7071067811865475};

int main(){
    float* m = dtwt_level_n(1, 2, h, 8, s);
    for(int i = 0; i < 8; i++)
        std::cout << m[i] << std::endl;

    return 0;
}

