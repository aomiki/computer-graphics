#ifndef UTILS_CUH
#define UTILS_CUH

#include "image_tools.h"
#include "cublas_v2.h"
#include <string>

static cublasStatus_t LAST_CUBLAS_ERROR = CUBLAS_STATUS_SUCCESS;
static cudaError_t LAST_CUDA_ERROR = cudaSuccess;
static std::string LAST_CUDA_ERROR_DESC = "";

void cuda_log(cublasStatus_t err);
void cuda_log(cudaError_t err);
matrix* transferMatrixToDevice(matrix* h_m);
void transferMatrixDataToHost(matrix* h_m, matrix* d_m, bool do_free = true);

#endif
