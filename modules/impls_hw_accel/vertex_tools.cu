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
__global__ void initVerticesArr(float** const c_vertices, float** c_offsets, float** rot_xyz_arr, vertices* verts, float* d_vertices_raw_arr_membuf, float rot_xyz[9], float offset[3])
{
    unsigned i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= verts->size)
        return;

    rot_xyz_arr[i] = rot_xyz;

    //copy vertices to c_vertices - they are simply going to be coefficients in gemv
    c_vertices[i] = d_vertices_raw_arr_membuf + i*3;
    c_vertices[i][0] = verts->x[i];
    c_vertices[i][1] = verts->y[i];
    c_vertices[i][2] = verts->z[i];
    
    //point each offset to vertex - so when gemv writes to offsets, it writes directly to vertex* vertices
    //point to the first element, each subsequent element is caculated using stride
    c_offsets[i] = verts->x + i;

    //replace vertex data with offset
    verts->x[i] = offset[0];
    verts->y[i] = offset[1];
    verts->z[i] = offset[2];
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

void vertex_transforms::rotateAndOffset(vertices* verts_transformed, vertices* verts, float offsets[3], float angles[3])
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

    vertices* d_verts; //array of vertices in our format
    const unsigned d_verts_bytes = sizeof(vertices);
    const unsigned d_verts_arrs_bytes = verts->size * sizeof(float);

    //Arrays of pointers for batched version of gemv
    float** d_vertices_raw_arr; //array of pointers to vertices in gemv format
    float** d_rot_xyz_arr; //array of pointers to the same matrix
    float** d_offsets_arr; //array of pointers to the same offsets
    const unsigned d_vert_arrays_batched_bytes = verts->size * sizeof(float*);
    
    float* d_offsets;
    const unsigned d_offsets_bytes = 3 * sizeof(float);

    float* d_vertices_raw_arr_membuf; //buffer for storing vertices in raw format
    const unsigned d_vertices_raw_arr_membuf_bytes = 3 * verts->size * sizeof(float);

    //Allocate one big raw memory chunk for all arrays
    char* d_gemv_arrays_membuf;
    cuda_log(cudaMalloc(&d_gemv_arrays_membuf, 
        d_verts_arrs_bytes * 3 +
        d_verts_bytes +                     //d_verts
        d_vert_arrays_batched_bytes * 3 +          //d_vertices_raw_arr, d_rot_xyz_arr, d_offsets_arr
        d_offsets_bytes +                 //d_offsets
        d_vertices_raw_arr_membuf_bytes  //d_vertices_raw_arr_membuf
    ));

    vertices h_verts_temp;
    h_verts_temp.size = verts->size;

    //map arrays from raw allocated memory
    unsigned offset = 0;

    d_vertices_raw_arr = (float**)(d_gemv_arrays_membuf + offset);
    offset += d_vert_arrays_batched_bytes;

    d_rot_xyz_arr = (float**)(d_gemv_arrays_membuf + offset);
    offset += d_vert_arrays_batched_bytes;

    d_verts = (vertices*)(d_gemv_arrays_membuf + offset);
    offset += d_verts_bytes;

    d_offsets_arr = (float**)(d_gemv_arrays_membuf + offset);
    offset += d_vert_arrays_batched_bytes;

    h_verts_temp.x = (float*)(d_gemv_arrays_membuf + offset);
    offset += d_verts_arrs_bytes;

    h_verts_temp.y = (float*)(d_gemv_arrays_membuf + offset);
    offset += d_verts_arrs_bytes;

    h_verts_temp.z = (float*)(d_gemv_arrays_membuf + offset);
    offset += d_verts_arrs_bytes;

    d_offsets =  (float*)(d_gemv_arrays_membuf + offset);
    offset += d_offsets_bytes;

    d_vertices_raw_arr_membuf = (float*)(d_gemv_arrays_membuf + offset);
    offset += d_vertices_raw_arr_membuf_bytes;

    //copy arrays data
    cuda_log(cudaMemcpy(h_verts_temp.x, verts->x, d_verts_arrs_bytes * 3, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_verts, &h_verts_temp, d_verts_bytes, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_offsets, offsets, d_offsets_bytes, cudaMemcpyHostToDevice));

    unsigned poly_total_blocksize = 32;
    if (verts->size >= 4480)
    {
        poly_total_blocksize = 128;
    }

    if (verts->size >= 8960)
    {
        poly_total_blocksize = 256;
    }

    if (verts->size >= 17920)
    {
        poly_total_blocksize = 512;
    }

    if (verts->size >= 35840)
    {
        poly_total_blocksize = 1024;
    }

    //initialize all arrays for batched gemv
    initVerticesArr<<<(unsigned)(verts->size/poly_total_blocksize + 1), poly_total_blocksize>>>(d_vertices_raw_arr, d_offsets_arr, d_rot_xyz_arr, d_verts, d_vertices_raw_arr_membuf, d_rot_xyz, d_offsets);
    cuda_log(cudaDeviceSynchronize());

    const float h_beta_gemv = 1;

    //Everything was for this moment
    cuda_log(cublasSgemvBatched(handle, CUBLAS_OP_N, 3, 3, &h_alpha, d_rot_xyz_arr, 3, d_vertices_raw_arr, 1, &h_beta_gemv, d_offsets_arr, verts->size, verts->size));
    cuda_log(cudaDeviceSynchronize());

    //Get our result back to RAM
    cuda_log(cudaMemcpy(verts_transformed, d_verts, sizeof(vertices), cudaMemcpyDeviceToHost));

    float* h_verts_arr = new float[verts->size * 3];
    cuda_log(cudaMemcpy(h_verts_arr, h_verts_temp.x, d_verts_arrs_bytes * 3, cudaMemcpyDeviceToHost));

    verts_transformed->x = h_verts_arr;
    verts_transformed->y = h_verts_arr + verts_transformed->size;
    verts_transformed->z = h_verts_arr + verts_transformed->size * 2;

    //Free GPU memory
    cuda_log(cudaFree(d_rot_membuf));
    cuda_log(cudaFree(d_gemv_arrays_membuf));
}
