# include <stdio.h>
#include <ATen/ATen.h>

// 定义 batch、head、dim 大小
int batch = 1024;
int head = 16;
int dim = 32;

int main(){
    printf("this is from unit");

    // 创建一个新的张量，并使用 zeros 函数初始化
    auto q = at::zeros({batch, head, dim}, at::kFloat);

    // 输出张量的大小
    std::cout << "Size of q: " << q.sizes() << std::endl;


    return 0;
}