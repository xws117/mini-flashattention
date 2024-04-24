#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>

const int M =16;
const int N =8;
const int K =16; 

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
	uint32_t fregC[4];

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

	// naive load
	int x =  threadIdx.x;
	if(x<K){
		for(int i=0;i<K;i++){
			smemA[x*K + i] = A[x*K + i]; // A[x][i]
		}
	}
	if(x<N){
		for(int i=0;i<K;i++){
			smemB[][]              // smem[][]
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

	// printf("======%u ===\n",fragA[0]);
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

	// ldmatrix for B
	start_id = 0;
	if (x<16){

	}


}


int main(){

	half A[M * K];
	half B[K * N];
	float C[M * N];
	float D[M * N];

	half* d_a;
	half* d_b;
	float* d_c;

	cudaMalloc(&d_a, sizeof(half) * M * K);
	cudaMalloc(&d_b, sizeof(half) * K * N);
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

	GPU_gemm<<<1,32>>>(d_a,d_b,d_c);

	cudaDeviceSynchronize();
	

	return 0;
}
