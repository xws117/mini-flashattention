#include <stdio.h>

__global__ void hello(){
    printf("this is from cuda");
}


int main(){
    printf("hello world");
    hello<<<1,3>>>();
    cudaDeviceSynchronize();
}