#include "cublas_v2.h"
#include "vertex_tools.h"
#include "utils.cuh"

/// @brief Initializes array of vertices, array of rotation matrices, array of offsets.
/// @brief Arrays of matrices and offsets are gonna contain the same memory address in all elements.
/// @param[out] c_vertices Array of vertices. Each element is an array of 3 elements.
/// @param[out] c_offsets Array of offsets. Each element is going to be equal to offset parameter.
/// @param[out] rot_xyz_arr Array of rotation matrices. Each element is goint to be equal to rot_xyz parameter.
/// @param[in] vertices Array of vertices.
/// @param[in] rot_xyz Pointer to rotation matrix.
/// @param[in] offset Pointer to offset vector.
/// @param[in] n_vert Number of vertices.
/// @return 
__global__ void kernel_initBatchPtrs(float** const vertices_batch_ptrs, float** offsets_batch_ptrs, float** rot_xyz_batch_ptrs, vertices* verts, vertices* verts_transformed, float rot_xyz[9], float offset[3])
{
    unsigned i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= verts->size)
        return;

    rot_xyz_batch_ptrs[i] = rot_xyz;

    //point each vertices_batch_ptrs to original vertices - they are simply going to be coefficients in gemv
    //point to the first element, each subsequent element is caculated using stride
    vertices_batch_ptrs[i] = verts->x + i;

    //point each offset to resulitng vertices buffer - so when gemv writes to offsets, it writes directly to vertex* resulting vertices
    //point to the first element, each subsequent element is caculated using stride
    offsets_batch_ptrs[i] = verts_transformed->x + i;

    //replace vertex data with offset
    verts_transformed->x[i] = offset[0];
    verts_transformed->y[i] = offset[1];
    verts_transformed->z[i] = offset[2];
}

__global__ void doubleArrToVertices(vertex* vertices, float** result, unsigned n_vert)
{
    unsigned i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= n_vert)
        return;

    //copy result back to vertex
    vertices[i].x = result[i][0];
    vertices[i].y = result[i][1];
    vertices[i].z = result[i][2];
}

cublasHandle_t handle;

vertex_transforms::vertex_transforms()
{
    cuda_log(cublasCreate(&handle));
    cuda_log(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

vertex_transforms::~vertex_transforms()
{
    cuda_log(cublasDestroy(handle));
}

void vertex_transforms::rotateAndOffset(vertices* d_verts_transformed, vertices* d_verts, unsigned n_verts, float offsets[3], float angles[3])
{
    if ((offsets[0] == 0) && (offsets[1] == 0) && (offsets[2] == 0) &&
        (angles[0] == 0) && (angles[1] == 0) && (angles[2] == 0))
    {
        return;
    }

    //1. 2xGEMM: MULTIPLY ROTATION MATRICES
    float cosx = cos(angles[0]), sinx = sin(angles[0]);
    float cosy = cos(angles[1]), siny = sin(angles[1]);
    float cosz = cos(angles[2]), sinz = sin(angles[2]);

    //column-major rotation matrices
    const unsigned rot_mat_size = 9 * sizeof(float);
    const float h_rot_buffer[9*3] = {
        //h_rot_x
        1, 0, 0,
        0, cosx, -sinx,
        0, sinx, cosx,

        //h_rot_y
        cosy, 0, -siny,
        0, 1, 0,
        siny, 0, cosy,

        //h_rot_z
        cosz, -sinz, 0,
        sinz, cosz, 0,
        0, 0, 1
    };

    //Copy rotation matrices to GPU in one go
    char* d_rot_membuf; //Memory on GPU to store 4 matrices
    cuda_log(cudaMalloc(&d_rot_membuf, rot_mat_size * 4));
    cuda_log(cudaMemcpy(d_rot_membuf, h_rot_buffer, rot_mat_size*3, cudaMemcpyHostToDevice));

    //Map matrices from raw memory
    float* d_rot_x = (float*) d_rot_membuf;
    float* d_rot_y = (float*)(d_rot_membuf + rot_mat_size);
    float* d_rot_z = (float*)(d_rot_membuf + rot_mat_size * 2);

    float* d_rot_xy   = (float*)(d_rot_membuf + rot_mat_size * 3);
    float* d_rot_xyz  = d_rot_x; //for the second gemm, not gonna need d_rot_x then

    //Coefficients
    const float h_alpha = 1;
    const float h_beta = 0;

    //rotation by x and y
    cuda_log(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &h_alpha, d_rot_x, 3, d_rot_y, 3, &h_beta, d_rot_xy, 3));

    //rotation by xy and z
    cuda_log(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &h_alpha, d_rot_xy, 3, d_rot_z, 3, &h_beta, d_rot_xyz, 3));

    //2. 1xGEMV: MULTIPLY MATRICES AND VECTORS, ADD OFFSET

    //Arrays of pointers for batched version of gemv
    float** d_vertices_batch_ptrs; //array of pointers to vertices in gemv format
    float** d_rot_xyz_batch_ptrs; //array of pointers to the same matrix
    float** d_offsets_batch_ptrs; //array of pointers to the same offsets
    const unsigned d_float_ptr_batch_bytes = n_verts * sizeof(float*);

    float* d_offsets;
    const unsigned d_offsets_bytes = 3 * sizeof(float);

    //Allocate one big raw memory chunk for all arrays
    char* d_batch_ptrs_membuf;
    cuda_log(cudaMalloc(&d_batch_ptrs_membuf,
        d_float_ptr_batch_bytes * 3 +    //d_vertices_batch_ptrs, d_rot_xyz_batch_ptrs, d_offsets_batch_ptrs
        d_offsets_bytes                 //d_offsets
    ));

    //map arrays from raw allocated memory
    unsigned offset = 0;

    d_vertices_batch_ptrs = (float**)(d_batch_ptrs_membuf + offset);
    offset += d_float_ptr_batch_bytes;

    d_rot_xyz_batch_ptrs = (float**)(d_batch_ptrs_membuf + offset);
    offset += d_float_ptr_batch_bytes;

    d_offsets_batch_ptrs = (float**)(d_batch_ptrs_membuf + offset);
    offset += d_float_ptr_batch_bytes;

    d_offsets =  (float*)(d_batch_ptrs_membuf + offset);
    offset += d_offsets_bytes;

    //copy arrays data
    cuda_log(cudaMemcpy(d_offsets, offsets, d_offsets_bytes, cudaMemcpyHostToDevice));

    unsigned poly_total_blocksize = 32;
    if (n_verts >= 4480)
    {
        poly_total_blocksize = 128;
    }

    if (n_verts >= 8960)
    {
        poly_total_blocksize = 256;
    }

    if (n_verts >= 17920)
    {
        poly_total_blocksize = 512;
    }

    if (n_verts >= 35840)
    {
        poly_total_blocksize = 1024;
    }

    //initialize all arrays for batched gemv
    kernel_initBatchPtrs<<<(unsigned)(n_verts/poly_total_blocksize + 1), poly_total_blocksize>>>(d_vertices_batch_ptrs, d_offsets_batch_ptrs, d_rot_xyz_batch_ptrs, d_verts, d_verts_transformed, d_rot_xyz, d_offsets);
    cuda_log(cudaDeviceSynchronize());

    const float h_beta_gemv = 1;

    //Everything was for this moment
    cuda_log(cublasSgemvBatched(handle, CUBLAS_OP_N, 3, 3, &h_alpha, d_rot_xyz_batch_ptrs, 3, d_vertices_batch_ptrs, n_verts, &h_beta_gemv, d_offsets_batch_ptrs, n_verts, n_verts));
    cuda_log(cudaDeviceSynchronize());

    //Free GPU memory
    cuda_log(cudaFree(d_rot_membuf));
    cuda_log(cudaFree(d_batch_ptrs_membuf));
}
