#include <cuda/std/limits>
#include <nvtx3/nvToolsExt.h>

#include "vertex_tools.h"
#include "image_draw_objects.h"
#include "utils.cuh"

__global__ void kernel_drawVertices(matrix* m, vertices* verts, unsigned char* components, float scaleX, float scaleY)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int comp = threadIdx.y;

    if (i >= verts->size)
        return;

    uint2 img_center { (unsigned)(m->width/2), (unsigned)(m->height/2) };

    int x = static_cast<int>(scaleX * verts->x[i] / verts->z[i] + img_center.x);
    int y = static_cast<int>(m->height - (scaleY * verts->y[i] / verts->z[i]  + img_center.y));

    if (x >= m->width || y >= m->height)
    {
        return;
    }

    m->get(x, y)[comp] = components[comp];
};



template<typename E>
void model_renderer::draw_vertices(matrix_color<E>* m, E vertex_color, float scaleX, float scaleY)
{
    unsigned max_block_length = (1024 / m->components_num);

    int block_length = min(max_block_length, n_verts);

    int block_num = (int)((n_verts / block_length)+1);

    dim3 blockSize(block_length, m->components_num);

    matrix* d_m;
    const unsigned d_m_bytes = sizeof(matrix);
    
    unsigned char* d_vertex_color;
    const unsigned d_vertex_color_bytes = m->components_num * sizeof(unsigned char);

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_m_bytes +
        d_vertex_color_bytes
    ));

    unsigned mem_offset = 0;

    d_m = (matrix*)(d_membuf + mem_offset);
    mem_offset += d_m_bytes;

    d_vertex_color = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_vertex_color_bytes;

    cuda_log(cudaMemcpy(d_m, (matrix*)m, sizeof(matrix), cudaMemcpyHostToDevice));

    unsigned char* h_vertex_color = new unsigned char[m->components_num];

    m->element_to_c_arr(h_vertex_color, vertex_color);
    cuda_log(cudaMemcpy(d_vertex_color, h_vertex_color, d_vertex_color_bytes, cudaMemcpyHostToDevice));

    kernel_drawVertices<<<block_num, blockSize>>>(d_m, d_verts_transformed, d_vertex_color, scaleX, scaleY);

    cuda_log(cudaDeviceSynchronize());

    cuda_log(cudaFree(d_membuf));
    
    delete h_vertex_color;
};

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__global__ void kernel_drawPolygon(matrix* m, uint2 screen_min, uint2 screen_max, unsigned char* polygon_color, float3 screen_v1, float3 screen_v2, float3 screen_v3, float* zbuffer = nullptr)
{
    uint2 i_curr {
        threadIdx.x + blockDim.x * blockIdx.x + screen_min.x,
        threadIdx.y + blockDim.y * blockIdx.y + screen_min.y
    };

    if (i_curr.x >= screen_max.x || i_curr.y >= screen_max.y)
        return;

    float3 baryc;
    get_barycentric_coords(baryc, i_curr, screen_v1, screen_v2, screen_v3);

    if ((baryc.x >= 0) && (baryc.y >= 0) && (baryc.z >= 0))
    {
        //z-buffer check, if available
        if (zbuffer != nullptr)
        {
            int interlaced_index = i_curr.y * m->width + i_curr.x;
            float curr_z = (baryc.x * screen_v1.z + baryc.y * screen_v2.z + baryc.z * screen_v3.z);
            atomicMinFloat(zbuffer + interlaced_index, curr_z);

            if (zbuffer[interlaced_index] == curr_z)
            {
                for (unsigned i = 0; i < m->components_num; i++)
                {
                    m->get(i_curr.x, i_curr.y)[i] = polygon_color[i];
                }
            }
        }
        else
        {
            for (unsigned i = 0; i < m->components_num; i++)
            {
                m->get(i_curr.x, i_curr.y)[i] = polygon_color[i];
            }
        }
    }
};

__global__ void kernel_fill(float* arr, unsigned size, float val)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;;

    if (i >= size)
        return;

    arr[i] = val;
}

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3)
{
    uint2 min { 0,0 };
    uint2 max { 0,0 };

    float3 d_v1 { v1.x, v1.y, v1.z };
    float3 d_v2 { v2.x, v2.y, v2.z };
    float3 d_v3 { v3.x, v3.y, v3.z };

    calc_triangle_boundaries(min, max, d_v1, d_v2, d_v3, *img);

    if (max.x <= min.x || max.y <= min.y)
    {
        return;
    }

    //CUDA-SPECIFIC
    unsigned poly_width = max.x - min.x;
    unsigned poly_height = max.y - min.y;

    unsigned blocksize_1d = 32; //32*32 = 1024 = max blocksize

    unsigned blocknum_x = (unsigned)((poly_width/blocksize_1d) +1);
    unsigned blocknum_y = (unsigned)((poly_height/blocksize_1d) +1);

    dim3 blocksize(blocksize_1d, blocksize_1d);
    dim3 blocknum(blocknum_x, blocknum_y);

    matrix* d_m;
    const unsigned d_m_bytes = sizeof(matrix);

    unsigned char* d_arr_interlaced;
    const unsigned d_arr_interlaced_bytes = img->size_interlaced();

    unsigned char* d_vals;
    const unsigned d_vals_bytes = img->components_num * sizeof(unsigned char);

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_m_bytes +
        d_arr_interlaced_bytes +
        d_vals_bytes
    ));

    unsigned mem_offset = 0;

    d_m = (matrix*)d_membuf;
    mem_offset += d_m_bytes;

    d_arr_interlaced = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_arr_interlaced_bytes;

    d_vals = (unsigned char*)(d_membuf += mem_offset);
    mem_offset += d_vals_bytes;
    
    transferMatrixToDevice(d_m, d_arr_interlaced, img);

    unsigned char* h_vals = new unsigned char[img->components_num];

    img->element_to_c_arr(h_vals, polyg_color);
    cuda_log(cudaMemcpy(d_vals, h_vals, d_vals_bytes, cudaMemcpyHostToDevice));

    kernel_drawPolygon<<<blocknum, blocksize>>>(d_m, min, max, d_vals, d_v1, d_v2, d_v3);

    cuda_log(cudaDeviceSynchronize());

    transferMatrixDataToHost(img, d_m, false);
    cuda_log(cudaFree(d_membuf));
    delete h_vals;
};

/// @brief Draw polygons and paint them using random values.
/// @param m image matrix
/// @param vertices raw vertices
/// @param polygons raw polygons
/// @param n_vert number of provided vertices 
/// @param n_poly number of provided polygons
/// @param scale how much to scale polygons
/// @param offset how much to offset polygons
/// @param curState used for random numbers generator
/// @param seed used for random numbers generator
__global__ void kernel_drawPolygonsFilled(matrix* m, vertices* verts, polygons* polys, float scaleX, float scaleY, float* zbuffer, unsigned char* c_polyg_color_buffer, unsigned char* modelColor)
{
    //polygon index
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= polys->size)
    {
        return;
    }

    unsigned poly_v1_i = polys->vertex_index1[i]-1;
    unsigned poly_v2_i = polys->vertex_index2[i]-1;
    unsigned poly_v3_i = polys->vertex_index3[i]-1;

    float3 poly_v1 { verts->x[poly_v1_i], verts->y[poly_v1_i], verts->z[poly_v1_i] };
    float3 poly_v2 { verts->x[poly_v2_i], verts->y[poly_v2_i], verts->z[poly_v2_i] };
    float3 poly_v3 { verts->x[poly_v3_i], verts->y[poly_v3_i], verts->z[poly_v3_i] };

    float3 poly_vec1;
    float3 poly_vec2;

    poly_vertices_to_vectors(poly_v1, poly_v2, poly_v3, poly_vec1, poly_vec2);

    float3 normal_vec { 0, 0, 0 };
    normal(normal_vec, poly_vec1, poly_vec2);

    float3 camera_vec { 0.0, 0.0, 1.0 };
    float d = dot(normal_vec, camera_vec);
    float viewing_angle_cosine = d/(length(normal_vec)*length(camera_vec));

    unsigned char* c_polyg_color = c_polyg_color_buffer + i * m->components_num;

    if (viewing_angle_cosine >= 0)
    {
        return;
    }

    for (size_t i = 0; i < m->components_num; i++)
    {
        c_polyg_color[i] = (unsigned char)(-1 * modelColor[i] * viewing_angle_cosine + 0.5);
    }

    uint2 img_center { (unsigned)(m->width/2), (unsigned)(m->height/2) };
 
    //retrieve polygon's vertices and scale them
    float3 screen_v1 {
        scaleX * poly_v1.x / poly_v1.z + img_center.x,
        m->height - (scaleY * poly_v1.y / poly_v1.z + img_center.y),
        poly_v1.z
    };
    float3 screen_v2 {
        scaleX * poly_v2.x / poly_v2.z + img_center.x,
        m->height - (scaleY * poly_v2.y / poly_v2.z + img_center.y),
        poly_v2.z
    };

    float3 screen_v3 {
        scaleX * poly_v3.x / poly_v3.z + img_center.x,
        m->height - (scaleY * poly_v3.y/ poly_v3.z + img_center.y),
        poly_v3.z
    };

    //printf("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%u\n", i, poly_v1.x, poly_v1.y, poly_v1.z, poly_v2.x, poly_v2.y, poly_v2.z, poly_v3.x, poly_v3.y, poly_v3.z, viewing_angle_cosine, c_polyg_color[0]);

    //calculate rectangular boundary of the triangle
    uint2 screen_min{0,0};
    uint2 screen_max{0,0};

    calc_triangle_boundaries(screen_min, screen_max, screen_v1, screen_v2, screen_v3, *m);

    if (screen_max.x <= screen_min.x || screen_max.y <= screen_min.y)
    {
        return;
    }

    //DRAW TRIANGLE

    //cuda indices
    unsigned poly_width = screen_max.x - screen_min.x;
    unsigned poly_height = screen_max.y - screen_min.y;

    unsigned total_submatrix_size = poly_width * poly_height;

    unsigned poly_total_blocksize = 32;
    if (total_submatrix_size >= 4480)
    {
        poly_total_blocksize = 128;
    }

    if (total_submatrix_size >= 8960)
    {
        poly_total_blocksize = 256;
    }

    if (total_submatrix_size >= 17920)
    {
        poly_total_blocksize = 512;
    }

    if (total_submatrix_size >= 35840)
    {
        poly_total_blocksize = 1024;
    }

    unsigned blocksize_1d = (unsigned)sqrtf(poly_total_blocksize);

    unsigned blocknum_x = (unsigned)((poly_width/blocksize_1d) +1);
    unsigned blocknum_y = (unsigned)((poly_height/blocksize_1d) +1);

    dim3 blocksize(blocksize_1d, blocksize_1d);
    dim3 blocknum(blocknum_x, blocknum_y);

    cudaStream_t s;
    cuda_log(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    kernel_drawPolygon<<<blocknum, blocksize, 0, s>>>(m, screen_min, screen_max, c_polyg_color, screen_v1, screen_v2, screen_v3, zbuffer);
    cuda_log(cudaGetLastError());
};

model_renderer::model_renderer(vertices *verts, polygons *polys)
{
    this->d_verts_transformed_membuf = nullptr;
    this->n_verts = verts->size;
    this->n_polys = polys->size;

    //Transfer geometry to GPU memory.
    const unsigned d_verts_bytes = sizeof(vertices);
    const unsigned d_verts_arrs_bytes = verts->size * sizeof(float);

    const unsigned d_polys_bytes = sizeof(polygons);
    const unsigned d_polys_arrs_bytes = polys->size * sizeof(unsigned);

    cuda_log(cudaMalloc(
        &d_original_geometry_membuf,
        d_verts_arrs_bytes * 3 +
        d_polys_arrs_bytes * 3 +
        d_verts_bytes +
        d_polys_bytes
    ));

    vertices h_verts_temp;
    h_verts_temp.size = verts->size;
    polygons h_polys_temp;
    h_polys_temp.size = polys->size;

    unsigned mem_offset = 0;

    h_verts_temp.x = (float*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_verts_arrs_bytes;

    h_verts_temp.y = (float*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_verts_arrs_bytes;

    h_verts_temp.z = (float*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_verts_arrs_bytes;

    h_polys_temp.vertex_index1 = (unsigned*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_polys_arrs_bytes;

    h_polys_temp.vertex_index2 = (unsigned*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_polys_arrs_bytes;

    h_polys_temp.vertex_index3 = (unsigned*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_polys_arrs_bytes;

    d_verts = (vertices*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_verts_bytes;

    d_polys = (polygons*)(d_original_geometry_membuf + mem_offset);
    mem_offset += d_polys_bytes;

    cuda_log(cudaMemcpy(h_verts_temp.x, verts->x, d_verts_arrs_bytes * 3, cudaMemcpyHostToDevice))
    cuda_log(cudaMemcpy(d_verts, &h_verts_temp, d_verts_bytes, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(h_polys_temp.vertex_index1, polys->vertex_index1, d_polys_arrs_bytes * 3, cudaMemcpyHostToDevice))
    cuda_log(cudaMemcpy(d_polys, &h_polys_temp, d_polys_bytes, cudaMemcpyHostToDevice));

    this->d_verts_transformed = d_verts;
}

model_renderer::~model_renderer()
{
    if (d_original_geometry_membuf != nullptr)
    {
        cudaFree(d_original_geometry_membuf);
    }
    if (d_verts_transformed_membuf != nullptr)
    {
        cudaFree(d_verts_transformed_membuf);
    }
}

void model_renderer::rotateAndOffset(float offsets[3], float angles[3], vertex_transforms* vt_transformer)
{
    const unsigned d_verts_bytes = sizeof(vertices);
    const unsigned d_verts_arrs_bytes = n_verts * sizeof(float);

    h_verts_transformed_d_copy.size = n_verts;

    //if space for transformed vertices wasn't allocated on GPU - allocate 
    if (d_verts_transformed == d_verts)
    {
        //Allocate raw memory
        cuda_log(cudaMalloc(
            &d_verts_transformed_membuf,
            d_verts_arrs_bytes * 3 +
            d_verts_bytes
        ));

        //partition raw space, get pointers to all arrays
        unsigned mem_offset = 0;

        h_verts_transformed_d_copy.x = (float*)(d_verts_transformed_membuf + mem_offset);
        mem_offset += d_verts_arrs_bytes;

        h_verts_transformed_d_copy.y = (float*)(d_verts_transformed_membuf + mem_offset);
        mem_offset += d_verts_arrs_bytes;

        h_verts_transformed_d_copy.z = (float*)(d_verts_transformed_membuf + mem_offset);
        mem_offset += d_verts_arrs_bytes;

        d_verts_transformed = (vertices*)(d_verts_transformed_membuf + mem_offset);
        mem_offset += d_verts_bytes;

        //copy pointers to GPU struct
        cuda_log(cudaMemcpy(d_verts_transformed, &h_verts_transformed_d_copy, d_verts_bytes, cudaMemcpyHostToDevice));
    }

    vt_transformer->rotateAndOffset(d_verts_transformed, d_verts, n_verts, offsets, angles);
}

template<typename E>
void model_renderer::draw_polygons<E>(matrix_color<E> *img, float scaleX, float scaleY, unsigned char* modelColor)
{
    nvtxRangeId_t nvtx_render_mark = nvtxRangeStartA("render_draw");

    //CUDA-SPECIFIC
    //Итерируемся по полигонам. На каждой такой итерации - получаем нужные вершины и конвертируем в координаты экрана.
    //Затем создаем квадрат в пределах этих вершин и итерируемся по каждому пикселю в квадрате, закрашивая либо не закрашивая по пути
    //основные индексы: i полигонов, x и y итерации по квадрату

    unsigned blocksize = 32;
    if (n_polys >= 4480)
    {
        blocksize = 128;
    }

    if (n_polys >= 8960)
    {
        blocksize = 256;
    }

    if (n_polys >= 17920)
    {
        blocksize = 512;
    }
/*
    if (n_polys >= 35840)
    {
        blocksize = 1024;
    }
*/

    nvtxRangeId_t nvtx_render_memory_to_mark = nvtxRangeStartA("render_draw_memory_to_gpu");
    float* d_zbuffer;
    const unsigned d_zbuffer_bytes = img->size() * sizeof(float);

    matrix* d_m;
    const unsigned d_m_bytes = sizeof(matrix);

    unsigned char* c_polyg_color_buffer;
    const unsigned c_polyg_color_buffer_bytes = n_polys * img->components_num;

    unsigned char* d_modelColor;
    const unsigned d_modelColor_bytes = img->components_num * sizeof(unsigned char);

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_zbuffer_bytes +
        d_m_bytes +
        c_polyg_color_buffer_bytes +
        d_modelColor_bytes
    ));

    unsigned mem_offset = 0;

    d_zbuffer = (float*)(d_membuf + mem_offset);
    mem_offset += d_zbuffer_bytes;

    d_m = (matrix*)(d_membuf + mem_offset);
    mem_offset += d_m_bytes;

    c_polyg_color_buffer = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += c_polyg_color_buffer_bytes;

    d_modelColor = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_modelColor_bytes;

    cuda_log(cudaMemcpy(d_m, (matrix*)img, d_m_bytes, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_modelColor, modelColor, d_modelColor_bytes, cudaMemcpyHostToDevice));

    nvtxRangeEnd(nvtx_render_memory_to_mark);

    unsigned zbuf_total_blocksize = 32;
    if (img->size() >= 4480)
    {
        zbuf_total_blocksize = 128;
    }

    if (img->size() >= 8960)
    {
        zbuf_total_blocksize = 256;
    }

    if (img->size() >= 17920)
    {
        zbuf_total_blocksize = 512;
    }

    if (img->size() >= 35840)
    {
        zbuf_total_blocksize = 1024;
    }

    kernel_fill<<<(unsigned)(img->size()/zbuf_total_blocksize +1), zbuf_total_blocksize>>>(d_zbuffer, img->size(), std::numeric_limits<float>().max());
    cuda_log(cudaGetLastError());

    unsigned blocknum = (unsigned)((n_polys / blocksize) + 1);

    //by default the limit on the number of simultaneous kernel launches is imposed 
    cuda_log(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, n_polys));

    kernel_drawPolygonsFilled<<<blocknum, blocksize>>>(d_m, d_verts_transformed, d_polys, scaleX, scaleY, d_zbuffer, c_polyg_color_buffer, d_modelColor);

    cuda_log(cudaGetLastError());
    cuda_log(cudaDeviceSynchronize());

    nvtxRangeId_t nvtx_render_memory_from_mark = nvtxRangeStartA("render_draw_memory_from_gpu");

    cuda_log(cudaFree(d_membuf));
    nvtxRangeEnd(nvtx_render_memory_from_mark);

    nvtxRangeEnd(nvtx_render_mark);
}


scene::scene()
{
    codec = new image_codec();
    vt_transformer = new vertex_transforms();
    img_matrix = nullptr;
}

scene::~scene()
{
    delete codec;
    delete vt_transformer;
    if (img_matrix != nullptr)
    {
        delete img_matrix;
    }
}

void scene::set_scene_params(unsigned width, unsigned height, ImageColorScheme colorScheme)
{
    if (img_matrix != nullptr)
    {
        if (width == img_matrix->width && height == img_matrix->height && colorScheme == this->colorScheme)
        {
            return;
        }

        if(colorScheme == this->colorScheme)
        {
            img_matrix->resize(width, height);
        }
        else
        {
            delete img_matrix;
            img_matrix = nullptr;
        }
        
    }

    if (img_matrix == nullptr)
    {
        this->colorScheme = colorScheme;
        switch (colorScheme)
        {
            case IMAGE_GRAY:
                img_matrix = new matrix_gray(width, height);
                break;
            case IMAGE_RGB:
                img_matrix = new matrix_rgb(width, height);
                break;
            default:
                break;
        }
    }
}

void scene::fill(unsigned char* color)
{
    img_matrix->fill(color);
}

void scene::encode(std::vector<unsigned char>& img_buffer)
{
    codec->encode(&img_buffer, img_matrix, colorScheme, 8);
}

void scene::draw_model_polygons(model_renderer& model, float scaleX, float scaleY, unsigned char* modelColor)
{
    switch (colorScheme)
    {
        case IMAGE_GRAY:
            model.draw_polygons((matrix_gray*)img_matrix, scaleX, scaleY, modelColor);
            break;
        case IMAGE_RGB:
            model.draw_polygons((matrix_rgb*)img_matrix, scaleX, scaleY, modelColor);
            break;
        default:
            break;
    }
}

void scene::draw_model_vertices(model_renderer& model, float scaleX, float scaleY, unsigned char* modelColor)
{
    switch (colorScheme)
    {
        case IMAGE_GRAY:
            model.draw_vertices((matrix_gray*)img_matrix, ((matrix_gray*)img_matrix)->c_arr_to_element(modelColor), scaleX, scaleY);
            break;
        case IMAGE_RGB:
            model.draw_vertices((matrix_rgb*)img_matrix, ((matrix_rgb*)img_matrix)->c_arr_to_element(modelColor), scaleX, scaleY);
            break;
        default:
            break;
    }
}

void scene::transform_model(model_renderer& model, float offsets[3], float angles[3])
{
    model.rotateAndOffset(offsets, angles, vt_transformer);
}

#include "_image_draw_objects_instances.h"
