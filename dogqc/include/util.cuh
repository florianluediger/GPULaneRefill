// Author of this file: Henning Funke

#pragma once

#define ALL_LANES 0xffffffff

// intialize an array as used e.g. for join hash tables
template<typename T>
__global__ void initArray ( T* array, T value, int num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
        array[i] = value;
    }
}


