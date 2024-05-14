#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>

const int M =16;
const int N =8;
const int K =16; 
const int block_size = 32;
void init_array(half* array,int len){
	for(int i=0;i<len;i++){
		array[i] = i;
	}

}

void cputest(half* A,half* B,float* C){
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			float sum = 0;
			for(int k=0;k<K;k++){
				//sum += A[i*K +k] * B[k*N +j];
				float a = __half2float(A[i*K +k]);
				float b = __half2float(B[k*N +j]);
				float c = a*b;
				sum += c;
			}
			C[i * N + j] = sum;
		}
	}
}



__global__ void GPU_gemm(half* A,half* B,float* C){

	__shared__ half smemA[M * K];
	__shared__ half smemB[K * N];

	uint32_t fragA[4];
	uint32_t fragB[2];
	// uint32_t fregC[4];

	// printf init A
	if (threadIdx.x==0){
		printf("===============GPU_init_A================\n");
		for(int i=0;i<M;i++){
			for(int j=0;j<K;j++){
				printf("%f ",__half2float(A[i*K + j]));
			}
			printf("\n");
		}
		printf("==========================================\n\n\n");
	}

    if (threadIdx.x==0){
		printf("===============GPU_init_B================\n");
		for(int i=0;i<N;i++){
			for(int j=0;j<K;j++){
				printf("%f ",__half2float(B[i*K + j]));
			}
			printf("\n");
		}
		printf("==========================================\n\n\n");
	}

	// naive load
	int x =  threadIdx.x;
	if(x<M){
		for(int i=0;i<K;i++){
			smemA[x*K + i] = A[x*K + i]; // A[x][i]
		}
	}
	if(x<N){
		for(int i=0;i<K;i++){
			smemB[x*K + i] = B[x*K + i];  // smem[][]
		}
	}

	//TODO 合并访存

	__syncthreads();

	if(x==0){
		printf("=============smemA =====================\n");
		for(int i=0;i<M;i++){
			for(int j=0;j<K;j++){
				printf("%f ",__half2float(smemA[i*K + j]));
			}
			printf("\n");
		}
        printf("=============smemA =====================\n\n");
	}

	// ldmatrix  for A
	int start_id = 0;
	if(x<16){
		start_id = x*16;
	}else{
		start_id = (x-16)*16 + 8;
	}
	asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n\t"
        : "=r"(fragA[0]), "=r"(fragA[1]), "=r"(fragA[2]), "=r"(fragA[3])
        : "l"(&smemA[start_id]));

	__syncthreads();

    // ldmatrix for B
    start_id = 0;
    if(x<8){
        start_id = x*16;
    }else{
        start_id = (x-8)*16 + 8;    
    }
    asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];\n\t"
        : "=r"(fragB[0]), "=r"(fragB[1])
        : "l"(&smemB[start_id]));
    __syncthreads();

    if(x==0){
		printf("=============smemB =====================\n");
		for(int i=0;i<N;i++){
			for(int j=0;j<K;j++){
				printf("%f ",__half2float(smemB[i*K + j]));
			}
			printf("\n");
		}
        printf("=============smemB =====================\n\n");
	}

	// printf("======%u ===\n",fragA[0]);   ============= A ====================
	half a1[2] = reinterpret_cast<half*>(&fragA[0])[0];
	(reinterpret_cast<float*>(&a1[0]))[0] = (reinterpret_cast<float*>(&fragA[0]))[0];
	half a2[2] = reinterpret_cast<half*>(&fragA[1])[0];
	(reinterpret_cast<float*>(&a2[0]))[0] = (reinterpret_cast<float*>(&fragA[1]))[0];
	half a3[2] = reinterpret_cast<half*>(&fragA[2])[0];
	(reinterpret_cast<float*>(&a3[0]))[0] = (reinterpret_cast<float*>(&fragA[2]))[0];
	half a4[2] = reinterpret_cast<half*>(&fragA[3])[0];
	(reinterpret_cast<float*>(&a4[0]))[0] = (reinterpret_cast<float*>(&fragA[3]))[0];
	
	printf("threadIdx  %d holds  reg[0]  %f %f reg[1]  %f %f reg[2]  %f %f reg[3]  %f %f\n",threadIdx.x,__half2float(a1[0]),__half2float(a1[1]),
	                    __half2float(a2[0]),__half2float(a2[1]),
						__half2float(a3[0]),__half2float(a3[1]),
						__half2float(a4[0]),__half2float(a4[1]));
	__syncthreads();

    // =======================================B===================================
	half b1[2] = reinterpret_cast<half*>(&fragB[0])[0];
	(reinterpret_cast<float*>(&b1[0]))[0] = (reinterpret_cast<float*>(&fragB[0]))[0];
	half b2[2] = reinterpret_cast<half*>(&fragB[1])[0];
	(reinterpret_cast<float*>(&b2[0]))[0] = (reinterpret_cast<float*>(&fragB[1]))[0];
	
	
	printf("[B] threadIdx  %d holds  reg[0]  %f %f reg[1]  %f %f \n",threadIdx.x,__half2float(b1[0]),__half2float(b1[1]),
	                    __half2float(b2[0]),__half2float(b2[1]));
	__syncthreads();

    //  mma
    float fragC[4]={0,0,0,0};
    asm(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    " { %0, %1, %2, %3 }, "
    " { %4, %5, %6, %7 }, "
    " { %8, %9 }, "
    " { %10, %11, %12, %13 };"
    :
    "=f"(fragC[0]), "=f"(fragC[1]), "=f"(fragC[2]), "=f"(fragC[3])
    :
    "r"(fragA[0]), "r"(fragA[1]), "r"(fragA[2]), "r"(fragA[3]),
    "r"(fragB[0]), "r"(fragB[1]),
    "f"(fragC[0]), "f"(fragC[1]), "f"(fragC[2]), "f"(fragC[3])
    );

    // printf("threadIdx  %d holds  %f");

    __syncthreads();

    printf("[C] threadIdx  %d holds  reg[0]  %f reg[1] %f reg[2]  %f reg[3] %f\n",threadIdx.x,fragC[0],fragC[1],fragC[2],fragC[3]);

}
__global__ void trans(half* A,half* B,int m, int n){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x<m && y<n){
        //A[x][y] = B[y][x];
        A[y*m + x]  = B[x*n + y];
    }
    __syncthreads();
    if (x==0 && y==0){
        printf("==================trans_B=============\n");
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                printf("%f ",__half2float(A[i*m + j]));
            }
            printf("\n");
        }
        printf("==================trans_B=============\n\n");
    }

}

int main(){

	half A[M * K];
	half B[K * N];
	float C[M * N];
	float D[M * N];

	half* d_a;
	half* d_b;
    half* d_b_trans;
	float* d_c;

	cudaMalloc(&d_a, sizeof(half) * M * K);
	cudaMalloc(&d_b, sizeof(half) * K * N);
    cudaMalloc(&d_b_trans, sizeof(half) * K * N);
	cudaMalloc(&d_c, sizeof(float) * M * N);

	init_array(A, M * K);
	init_array(B, K * N);
	//init_array(C, M * N);

	printf("===============cpu_init_A================\n");
	for(int i=0;i<M;i++){
		for(int j=0;j<K;j++){
			printf("%f ",__half2float(A[i*K + j]));
		}
		printf("\n");
	}
	printf("==========================================\n\n\n");


	cudaMemcpy(d_a,A, M * K * sizeof(half),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,B ,K * N * sizeof(half),cudaMemcpyHostToDevice);

    dim3 grid( (K+block_size-1)/block_size, (N+block_size-1)/block_size );
    dim3 block(block_size,block_size);
    trans<<<grid,block>>>(d_b_trans,d_b,K,N);

	// for(int i=0;i<10;i++){
	// 	printf("%f %f %f\n",__half2float(A[i]),__half2float(A[i]),__half2float(A[i]));
	// }

	// 使用CPU进行计算
	cputest(A,B,C);

	// 
	printf("======================cpu_computed_C=========================\n");
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			printf(" %f ",C[i*N + j]);
		}
		printf("\n");
	}
	printf("======================cpu_computed_C=========================\n\n\n");

	GPU_gemm<<<1,32>>>(d_a,d_b_trans,d_c);

	cudaDeviceSynchronize();
	

	return 0;
}