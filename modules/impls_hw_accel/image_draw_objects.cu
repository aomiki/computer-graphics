#include <cuda/std/limits>

#include "vertex_tools.h"
#include "image_draw_objects.h"
#include "utils.cuh"

__global__ void kernel_drawVertices(matrix* m, vertex* vertices, unsigned int n_vertices, unsigned char* components, int scale, int offset)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int comp = threadIdx.y;

    if (i >= n_vertices)
        return;

   int x = static_cast<int>(scale * vertices[i].x + offset);
   int y = static_cast<int>(m->height - (scale * vertices[i].y + offset));

    m->get(x, y)[comp] = components[comp];
};

template<typename E>
void draw_vertices(matrix_color<E>* m, std::vector<vertex>* vertices, E vertex_color, int scale, int offset)
{
    size_t max_block_length = (1024 / m->components_num);

    int block_length = min(max_block_length, vertices->size());

    int block_num = (int)((vertices->size() / block_length)+1);

    dim3 blockSize(block_length, m->components_num);

    vertex* d_vector;
    const unsigned d_vector_bytes = vertices->size() * sizeof(vertex);

    matrix* d_m;
    const unsigned d_m_bytes = sizeof(matrix);

    unsigned char* d_arr_interlaced;
    const unsigned d_arr_interlaced_bytes = m->size_interlaced();
    
    unsigned char* d_vals;
    const unsigned d_vals_bytes = m->components_num * sizeof(unsigned char);

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_vector_bytes +
        d_m_bytes +
        d_arr_interlaced_bytes +
        d_vals_bytes
    ));

    unsigned mem_offset = 0;

    d_vector = (vertex*)(d_membuf + mem_offset);
    mem_offset += d_vector_bytes;

    d_m = (matrix*)(d_membuf + mem_offset);
    mem_offset += d_m_bytes;

    d_arr_interlaced = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_arr_interlaced_bytes;

    d_vals = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_vals_bytes;

    transferMatrixToDevice(d_m, d_arr_interlaced, m);

    unsigned char* h_vals = new unsigned char[m->components_num];

    m->element_to_c_arr(h_vals, vertex_color);
    cuda_log(cudaMemcpy(d_vals, h_vals, d_vals_bytes, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_vector, vertices->data(), d_vector_bytes, cudaMemcpyHostToDevice));

    kernel_drawVertices<<<block_num, blockSize>>>(d_m, d_vector, vertices->size(), d_vals, scale, offset);

    cuda_log(cudaDeviceSynchronize());

    transferMatrixDataToHost(m, d_m, false);

    cuda_log(cudaFree(d_membuf));
    
    delete h_vals;
};

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void kernel_drawPolygon(matrix* m, matrix_coord screen_min, matrix_coord screen_max, unsigned char* polygon_color, vertex screen_v1, vertex screen_v2, vertex screen_v3, double* zbuffer = nullptr)
{
    matrix_coord i_curr(
        threadIdx.x + blockDim.x * blockIdx.x + screen_min.x,
        threadIdx.y + blockDim.y * blockIdx.y + screen_min.y
    );

    if (i_curr.x >= screen_max.x || i_curr.y >= screen_max.y)
        return;

    vertex baryc = get_barycentric_coords(i_curr, screen_v1, screen_v2, screen_v3);

    if ((baryc.x >= 0) && (baryc.y >= 0) && (baryc.z >= 0))
    {
        //z-buffer check, if available
        if (zbuffer != nullptr)
        {
            int interlaced_index = i_curr.y * m->width + i_curr.x;
            double curr_z = (baryc.x * screen_v1.z + baryc.y * screen_v2.z + baryc.z * screen_v3.z);
            atomicMin_double(zbuffer + interlaced_index, curr_z);

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

__global__ void kernel_fill(double* arr, unsigned size, double val)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;;

    if (i >= size)
        return;

    arr[i] = val;
}

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3)
{
    matrix_coord min(0,0);
    matrix_coord max(0,0);

    calc_triangle_boundaries(min, max, v1, v2, v3, *img);

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

    kernel_drawPolygon<<<blocknum, blocksize>>>(d_m, matrix_coord(min.x, min.y), matrix_coord(max.x, max.y), d_vals, v1, v2, v3);

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
__global__ void kernel_drawPolygonsFilled(matrix* m, vertex* vertices, polygon* polygons, unsigned n_vert, unsigned n_poly, double scaleX, double scaleY, double* zbuffer, unsigned char* c_polyg_color_buffer, unsigned char* modelColor)
{
    //polygon index
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= n_poly)
    {
        return;
    }

    polygon* curr_poly = polygons + i;

    vertex* poly_v1 = vertices + (curr_poly->vertex_index1-1);
    vertex* poly_v2 = vertices + (curr_poly->vertex_index2-1);
    vertex* poly_v3 = vertices + (curr_poly->vertex_index3-1);

    vertex poly_vec1;
    vertex poly_vec2;

    poly_vertices_to_vectors(*poly_v1, *poly_v2, *poly_v3, poly_vec1, poly_vec2);

    vertex n = normal(poly_vec1, poly_vec2);

    vertex camera_vec(0.0, 0.0, 1.0);
    double d = dot(n, camera_vec);
    double viewing_angle_cosine = d/(length(n)*length(camera_vec));

    unsigned char* c_polyg_color = c_polyg_color_buffer + i * m->components_num;

    if (viewing_angle_cosine >= 0)
    {
        return;
    }

    for (size_t i = 0; i < m->components_num; i++)
    {
        c_polyg_color[i] = (unsigned char)(-1 * modelColor[i] * viewing_angle_cosine + 0.5);
    }

    matrix_coord img_center((unsigned)(m->width/2), (unsigned)(m->height/2));
 
    //retrieve polygon's vertices and scale them
    vertex screen_v1{
        scaleX * poly_v1->x / poly_v1->z + img_center.x,
        m->height - (scaleY * poly_v1->y / poly_v1->z + img_center.y),
        poly_v1->z
    };
    vertex screen_v2{
        scaleX * poly_v2->x / poly_v2->z + img_center.x,
        m->height - (scaleY * poly_v2->y / poly_v2->z + img_center.y),
        poly_v2->z
    };
    vertex screen_v3{
        scaleX * poly_v3->x / poly_v3->z + img_center.x,
        m->height - (scaleY * poly_v3->y/ poly_v3->z + img_center.y),
        poly_v3->z
    };

    //printf("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%u\n", i, poly_v1.x, poly_v1.y, poly_v1.z, poly_v2.x, poly_v2.y, poly_v2.z, poly_v3.x, poly_v3.y, poly_v3.z, viewing_angle_cosine, c_polyg_color[0]);

    //calculate rectangular boundary of the triangle
    matrix_coord screen_min(0,0);
    matrix_coord screen_max(0,0);

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

template <typename E>
inline void draw_polygons_filled(matrix_color<E> *img, std::vector<vertex> *vertices, std::vector<polygon> *polygons, double scaleX, double scaleY, unsigned char* modelColor)
{
    //CUDA-SPECIFIC
    //Итерируемся по полигонам. На каждой такой итерации - получаем нужные вершины и конвертируем в координаты экрана.
    //Затем создаем квадрат в пределах этих вершин и итерируемся по каждому пикселю в квадрате, закрашивая либо не закрашивая по пути
    //основные индексы: i полигонов, x и y итерации по квадрату
    
    unsigned blocksize = 32;
    if (polygons->size() >= 4480)
    {
        blocksize = 128;
    }

    if (polygons->size() >= 8960)
    {
        blocksize = 256;
    }

    if (polygons->size() >= 17920)
    {
        blocksize = 512;
    }
/*
    if (polygons->size() >= 35840)
    {
        blocksize = 1024;
    }
*/

    double* d_zbuffer;
    const unsigned d_zbuffer_bytes = img->size() * sizeof(double);

    vertex* d_vertices;
    const unsigned d_vertices_bytes = vertices->size() * sizeof(vertex);

    polygon* d_polygons;
    const unsigned d_polygons_bytes = polygons->size() * sizeof(polygon);

    matrix* d_m;
    const unsigned d_m_bytes = sizeof(matrix);

    unsigned char* d_arr_interlaced;
    const unsigned d_arr_interlaced_bytes = img->size_interlaced();

    unsigned char* c_polyg_color_buffer;
    const unsigned c_polyg_color_buffer_bytes = polygons->size() * img->components_num;

    unsigned char* d_modelColor;
    const unsigned d_modelColor_bytes = img->components_num * sizeof(unsigned char);

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_zbuffer_bytes +
        d_vertices_bytes +
        d_polygons_bytes +
        d_m_bytes + 
        d_arr_interlaced_bytes + 
        c_polyg_color_buffer_bytes +
        d_modelColor_bytes
    ));

    unsigned mem_offset = 0;

    d_zbuffer = (double*)(d_membuf + mem_offset);
    mem_offset += d_zbuffer_bytes;

    d_vertices = (vertex*)(d_membuf + mem_offset);
    mem_offset += d_vertices_bytes;

    d_polygons = (polygon*)(d_membuf + mem_offset);
    mem_offset += d_polygons_bytes;

    d_m = (matrix*)(d_membuf + mem_offset);
    mem_offset += d_m_bytes;

    d_arr_interlaced = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_arr_interlaced_bytes;

    c_polyg_color_buffer = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += c_polyg_color_buffer_bytes;

    d_modelColor = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_modelColor_bytes;

    transferMatrixToDevice(d_m, d_arr_interlaced, img);

    cuda_log(cudaMemcpy(d_vertices, vertices->data(), d_vertices_bytes, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_polygons, polygons->data(), d_polygons_bytes, cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_modelColor, modelColor, d_modelColor_bytes, cudaMemcpyHostToDevice));

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

    kernel_fill<<<(unsigned)(img->size()/zbuf_total_blocksize +1), zbuf_total_blocksize>>>(d_zbuffer, img->size(), std::numeric_limits<double>().max());
    cuda_log(cudaGetLastError());

    unsigned blocknum = (unsigned)((polygons->size() / blocksize) + 1);

    //by default the limit on the number of simultaneous kernel launches is imposed 
    cuda_log(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, polygons->size()));

    kernel_drawPolygonsFilled<<<blocknum, blocksize>>>(d_m, d_vertices, d_polygons, vertices->size(), polygons->size(), scaleX, scaleY, d_zbuffer, c_polyg_color_buffer, d_modelColor);
    cuda_log(cudaGetLastError());
    cuda_log(cudaDeviceSynchronize());

    transferMatrixDataToHost(img, d_m, false);

    cudaFree(d_membuf);
}

#include "_image_draw_objects_instances.h"
