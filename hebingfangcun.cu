#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // 数据规模
#define THREADS_PER_BLOCK 64

// 核函数：从全局内存读取数据到寄存器，并进行简单操作
__global__ void globalToRegister(uint4 *d_out, const char *d_in) {
    // 计算线程全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保不越界
    if (tid * sizeof(uint4) < N) {
        // 从全局内存读取数据到寄存器
        uint4 data;
        char *dataPtr = (char*)&data;

        for (int i = 0; i < sizeof(uint4); ++i) {
            dataPtr[i] = d_in[tid * sizeof(uint4) + i];
        }

        // 简单操作，例如，将每个元素加1
        data.x += 1;
        data.y += 1;
        data.z += 1;
        data.w += 1;

        // 将结果写回全局内存
        d_out[tid] = data;
    }
}

int main() {
    // 分配主机内存
    char *h_in = new char[N];
    uint4 *h_out = new uint4[N / sizeof(uint4)];

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<char>(i % 256);  // 简单初始化
    }

    // 分配设备内存
    char *d_in = nullptr;
    uint4 *d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(char));
    cudaMalloc(&d_out, (N / sizeof(uint4)) * sizeof(uint4));

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_in, h_in, N * sizeof(char), cudaMemcpyHostToDevice);

    // 计算网格和块的尺寸
    int blocks = (N / sizeof(uint4) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // 启动核函数
    globalToRegister<<<blocks, THREADS_PER_BLOCK>>>(d_out, d_in);

    // 将结果从设备复制到主机
    cudaMemcpy(h_out, d_out, (N / sizeof(uint4)) * sizeof(uint4), cudaMemcpyDeviceToHost);

    // 输出结果（仅用于调试）
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_out[" << i << "] = {"
                  << h_out[i].x << ", "
                  << h_out[i].y << ", "
                  << h_out[i].z << ", "
                  << h_out[i].w << "}" << std::endl;
    }

    // 释放设备内存
    cudaFree(d_in);
    cudaFree(d_out);

    // 释放主机内存
    delete[] h_in;
    delete[] h_out;

    return 0;
}
