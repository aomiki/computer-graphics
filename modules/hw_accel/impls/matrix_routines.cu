#include "matrix_routines.h"
#include <stdio.h>

cudaError_t LAST_CUDA_ERROR = cudaSuccess;

static void cuda_log(cudaError_t err)
{
    LAST_CUDA_ERROR = err;
}

__global__ void kernel_fillInterlaced(unsigned char* arr, unsigned int n_arr, unsigned char* components, unsigned int n_comp)
{
    int t_id = (threadIdx.x + blockDim.x * blockIdx.x) * n_comp;
    t_id += threadIdx.y; //offset for components of RGB

    if (t_id >= n_arr)
        return;
    
   // printf("Index: %d (Block (%d, %d), Thread (%d, %d)), Component element: %d\n", t_id, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, components[threadIdx.y]);

    arr[t_id] = components[threadIdx.y];
}

void fillInterlaced(unsigned char *arr, const unsigned int n_arr, unsigned char* components, const unsigned int n_comps)
{
    int n_arr_1comp = n_arr/n_comps; //components are calculated separatly
    int max_threads_per_row = (int)(1024/n_comps); //gonna have number of rows in a block = number of components

    int threads_per_row = n_arr_1comp < max_threads_per_row? n_arr_1comp : max_threads_per_row; //what's bigger: arr size or max block size?
    dim3 blockSize(threads_per_row, n_comps);

    float f_blockNum = n_arr_1comp/(float)threads_per_row;
    int blockNum = f_blockNum == (int)f_blockNum?  (int)f_blockNum : (int)(f_blockNum+1); //if threads fit in blocks perfectly, just round. Otherwise add 1.

    unsigned char* d_arr;
    unsigned char* d_vals;
    cuda_log(cudaMalloc(&d_arr, n_arr * sizeof(unsigned char)));
    cuda_log(cudaMalloc(&d_vals, n_comps * sizeof(unsigned char)));
    cuda_log(cudaMemcpy(d_vals, components, n_comps*sizeof(unsigned char), cudaMemcpyHostToDevice));

    kernel_fillInterlaced<<<blockNum, blockSize>>>(d_arr, n_arr, d_vals, n_comps);
    cuda_log(cudaDeviceSynchronize());
    fflush(stdout);

    cuda_log(cudaMemcpy(arr, d_arr, n_arr*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cuda_log(cudaFree(d_vals));
    cuda_log(cudaFree(d_arr));
}
