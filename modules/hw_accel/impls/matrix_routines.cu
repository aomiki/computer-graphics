#include "matrix_routines.h"
#include <stdio.h>
#include <image_tools.h>

cudaError_t LAST_CUDA_ERROR = cudaSuccess;

static void cuda_log(cudaError_t err)
{
    LAST_CUDA_ERROR = err;
}

__global__ void kernel_fillInterlaced(matrix* m, unsigned char* components)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= m->width || y >= m->height)
        return;

   // printf("Index: %d (Block (%d, %d), Thread (%d, %d)), Component element: %d\n", t_id, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, components[threadIdx.y]);

    m->get(x, y)[threadIdx.z] = components[threadIdx.z];
}

matrix* transferMatrixToDevice(matrix* h_m)
{
    matrix* d_m;
    unsigned char* h_arr = h_m->get_arr_interlaced();
    unsigned char* d_arr;

    cuda_log(cudaMalloc(&d_m,  sizeof(matrix)));
    cuda_log(cudaMalloc(&d_arr, h_m->size_interlaced()));
    cuda_log(cudaMemcpy(d_arr, h_arr, h_m->size_interlaced(), cudaMemcpyHostToDevice));
    h_m->set_arr_interlaced(d_arr);

    cuda_log(cudaMemcpy(d_m, h_m, sizeof(matrix), cudaMemcpyHostToDevice));

    h_m->set_arr_interlaced(h_arr);

    return d_m;
}

void transferMatrixDataToHost(matrix* h_m, matrix* d_m, bool do_free = true)
{
    unsigned char* h_arr = h_m->get_arr_interlaced();

    cuda_log(cudaMemcpy(h_m, d_m, sizeof(matrix), cudaMemcpyDeviceToHost));
    cuda_log(cudaMemcpy(h_arr, h_m->get_arr_interlaced(), h_m->size_interlaced(), cudaMemcpyDeviceToHost));

    if (do_free)
    {
        cudaFree(h_m->get_arr_interlaced());
        cudaFree(d_m);
    }

    h_m->set_arr_interlaced(h_arr);
}

/// @brief Расположение элементов будет - по оси x = ось x матрицы, по оси y = ось y матрицы, по оси z = компоненты каждого элемента матрицы
/// @param m 
/// @param components Значение которое нужно установить
void fillInterlaced(matrix* m, unsigned char* components)
{
    int blocksize_2d = (int)(1024/m->components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(m->width / blocksize_1d + 1);
    int blocksnum_y = (int)(m->height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, m->components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    matrix* d_m = transferMatrixToDevice(m);

    unsigned char* d_vals;
    cuda_log(cudaMalloc(&d_vals, m->components_num * sizeof(unsigned char)));
    cuda_log(cudaMemcpy(d_vals, components, m->components_num * sizeof(unsigned char), cudaMemcpyHostToDevice));

    kernel_fillInterlaced<<<gridSize, blockSize>>>(d_m, d_vals);
    cuda_log(cudaDeviceSynchronize());
    fflush(stdout);

    transferMatrixDataToHost(m, d_m);

    cuda_log(cudaFree(d_vals));
}
