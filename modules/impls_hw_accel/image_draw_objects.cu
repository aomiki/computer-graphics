#include <curand.h>
#include <curand_kernel.h>
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

    matrix* d_m = transferMatrixToDevice(m);

    unsigned char* d_vals;
    unsigned char* h_vals = new unsigned char[m->components_num];
    vertex* d_vector;
    cuda_log(cudaMalloc(&d_vals, m->components_num * sizeof(unsigned char)));
    cuda_log(cudaMalloc(&d_vector, vertices->size() * sizeof(vertex)));

    m->element_to_c_arr(h_vals, vertex_color);
    cuda_log(cudaMemcpy(d_vals, h_vals, m->components_num * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_vector, vertices->data(), vertices->size() * sizeof(vertex), cudaMemcpyHostToDevice));

    kernel_drawVertices<<<block_num, blockSize>>>(d_m, d_vector, vertices->size(), d_vals, scale, offset);

    cuda_log(cudaDeviceSynchronize());

    transferMatrixDataToHost(m, d_m);

    cuda_log(cudaFree(d_vals));
    cuda_log(cudaFree(d_vector));
    
    delete h_vals;
};

__global__ void kernel_drawPolygon(matrix* m, matrix_coord min, matrix_coord max, unsigned char* polygon_color, vertex v1, vertex v2, vertex v3, long long* zbuffer = nullptr)
{
    matrix_coord curr(
        threadIdx.x + blockDim.x * blockIdx.x + min.x,
        threadIdx.y + blockDim.y * blockIdx.y + min.y
    );
    unsigned z = threadIdx.z;

    if (curr.x >= max.x || curr.y >= max.y)
        return;

    vertex baryc = get_barycentric_coords(curr, v1, v2, v3);

    if ((baryc.x >= 0) && (baryc.y >= 0) && (baryc.z >= 0))
    {
        //z-buffer check, if available
        if (zbuffer != nullptr)
        {
            int interlaced_index = curr.y * m->width +curr.x;
            long long curr_z = (long long) ::cuda::std::numeric_limits<double>::max()/2 * (baryc.x * v1.z + baryc.y * v2.z + baryc.z * v3.z);
            atomicMin(zbuffer + interlaced_index, curr_z);

            if (zbuffer[interlaced_index] == curr_z)
            {
                m->get(curr.x, curr.y)[z] = polygon_color[z];
                //printf("M(%d, %d)\tV1(%f, %f, %f)\tV2(%f, %f, %f)\tV3(%f, %f, %f)\tcolor: %d\tdraw\n", curr.x, curr.y, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, polygon_color[0]);
            }
            else
            {
                //printf("M(%d, %d)\tV1(%f, %f, %f)\tV2(%f, %f, %f)\tV3(%f, %f, %f)\tcolor: %d\tskip\n", curr.x, curr.y, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, polygon_color[0]);
            }
        }
        else
        {
            //printf("!!! M(%d, %d)\tV1(%f, %f, %f)\tV2(%f, %f, %f)\tV3(%f, %f, %f)\tcolor: %d\tdraw\n", curr.x, curr.y, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, polygon_color[0]);
            m->get(curr.x, curr.y)[z] = polygon_color[z];
        }
    }
};

__global__ void kernel_fill(long long* arr, long long val)
{
    int i = blockIdx.x;

    arr[i] = val;
}

__host__ __device__ void calc_triangle_boundaries(matrix_coord& min_coord, matrix_coord& max_coord, vertex v1, vertex v2, vertex v3, matrix& m)
{
    double xmin = min(min(v1.x, v2.x), v3.x);
    double ymin = min(min(v1.y, v2.y), v3.y);

    double xmax = max(max(v1.x, v2.x), v3.x)+1;
    double ymax = max(max(v1.y, v2.y), v3.y)+1;

    //crop to img boundaries
    if (xmin < 0)
    {
        xmin = 0;
    }
    else if(xmax > m.width)
    {
        xmax = m.width;
    }

    if (ymin < 0)
    {
        ymin = 0;
    }
    else if (ymax > m.height)
    {
        ymax = m.height;
    }

    min_coord.x = round(xmin);
    max_coord.x = round(xmax);

    min_coord.y = round(ymin);
    max_coord.y = round(ymax);
};

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

    unsigned blocksize_xy = (unsigned)(1024 / img->components_num);
    unsigned blocksize_1d = (unsigned)sqrt(blocksize_xy);

    unsigned blocknum_x = (unsigned)((poly_width/blocksize_1d) +1);
    unsigned blocknum_y = (unsigned)((poly_height/blocksize_1d) +1);

    dim3 blocksize(blocksize_1d, blocksize_1d, img->components_num);
    dim3 blocknum(blocknum_x, blocknum_y);

    matrix* d_m = transferMatrixToDevice(img);

    unsigned char* d_vals;
    unsigned char* h_vals = new unsigned char[img->components_num];
    cuda_log(cudaMalloc(&d_vals, img->components_num * sizeof(unsigned char)));

    img->element_to_c_arr(h_vals, polyg_color);
    cuda_log(cudaMemcpy(d_vals, h_vals, img->components_num * sizeof(unsigned char), cudaMemcpyHostToDevice));

    kernel_drawPolygon<<<blocknum, blocksize>>>(d_m, matrix_coord(min.x, min.y), matrix_coord(max.x, max.y), d_vals, v1, v2, v3);

    cuda_log(cudaDeviceSynchronize());

    transferMatrixDataToHost(img, d_m);
    cuda_log(cudaFree(d_vals));
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
__global__ void kernel_drawPolygonsFilled(matrix* m, vertex* vertices, polygon* polygons, unsigned n_vert, unsigned n_poly, unsigned scale, unsigned offset, curandState* curState, unsigned long seed, long long* zbuffer, unsigned char* polyColors)
{
    //polygon index
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
/*
    curand_init(seed, i, 0, curState);
  */  
    //get random color
    unsigned char* c_polyg_color = (unsigned char*)malloc(m->components_num);
    /*
    for (unsigned i = 0; i < m->components_num; i++)
    {
        float RANDOM = curand_uniform(curState);

        unsigned int rand_num = truncf(255 * RANDOM);
        c_polyg_color[i] = rand_num;
    }
*/
    vertex poly_v1(
        vertices[polygons[i].vertex_index1-1].x,
        vertices[polygons[i].vertex_index1-1].y,
        vertices[polygons[i].vertex_index1-1].z
    );
    vertex poly_v2(
        vertices[polygons[i].vertex_index2-1].x,
        vertices[polygons[i].vertex_index2-1].y,
        vertices[polygons[i].vertex_index2-1].z
    );
    vertex poly_v3{
        vertices[polygons[i].vertex_index3-1].x,
        vertices[polygons[i].vertex_index3-1].y,
        vertices[polygons[i].vertex_index3-1].z
    };

    vertex poly_vec1;
    vertex poly_vec2;

    poly_vertices_to_vectors(poly_v1, poly_v2, poly_v3, poly_vec1, poly_vec2);

    vertex n = normal(poly_vec1, poly_vec2);

    vertex camera_vec(0.0, 0.0, 1.0);
    double d = dot(n, camera_vec);
    double viewing_angle_cosine = d/(length(n)*length(camera_vec));

    if (viewing_angle_cosine > -0.9)
    {
        free(c_polyg_color);
        return;
    }

    c_polyg_color[0] = (unsigned char)(-255.0 * viewing_angle_cosine + 0.5);
    polyColors[i] = c_polyg_color[0];

    //retrieve polygon's vertices and scale them
    vertex v1{
        scale * poly_v1.x + offset,
        m->height - (scale * poly_v1.y + offset),
        poly_v1.z
    };
    vertex v2{
        scale * poly_v2.x + offset,
        m->height - (scale * poly_v2.y + offset),
        poly_v2.z
    };
    vertex v3{
        scale * poly_v3.x + offset,
        m->height - (scale * poly_v3.y + offset),
        poly_v3.z
    };

    printf("%d V1(%f,%f,%f)\tV2(%f,%f,%f)\tV3(%f,%f,%f)\t%u\n", i, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, c_polyg_color[0]);

    //calculate rectangular boundary of the triangle
    matrix_coord min(0,0);
    matrix_coord max(0,0);

    calc_triangle_boundaries(min, max, v1, v2, v3, *m);

    if (max.x <= min.x || max.y <= min.y)
    {
        free(c_polyg_color);
        return;
    }

    //DRAW TRIANGLE

    //cuda indices
    unsigned poly_width = max.x - min.x;
    unsigned poly_height = max.y - min.y;

    unsigned blocksize_xy = (unsigned)(1024 / m->components_num);
    unsigned blocksize_1d = (unsigned)sqrtf(blocksize_xy);

    unsigned blocknum_x = (unsigned)((poly_width/blocksize_1d) +1);
    unsigned blocknum_y = (unsigned)((poly_height/blocksize_1d) +1);

    dim3 blocksize(blocksize_1d, blocksize_1d, m->components_num);
    dim3 blocknum(blocknum_x, blocknum_y);

    kernel_drawPolygon<<<blocknum, blocksize>>>(m, min, max, c_polyg_color, v1, v2, v3, zbuffer);

    free(c_polyg_color);
};

template <typename E>
inline void draw_polygons_filled(matrix_color<E> *img, std::vector<vertex> *vertices, std::vector<polygon> *polygons, int scale, int offset)
{
    curandState* randState;
    cudaMalloc (&randState, sizeof(curandState));

    //CUDA-SPECIFIC
    //Итерируемся по полигонам. На каждой такой итерации - получаем нужные вершины и конвертируем в координаты экрана.
    //Затем создаем квадрат в пределах этих вершин и итерируемся по каждому пикселю в квадрате, закрашивая либо не закрашивая по пути
    //основные индексы: i полигонов, x и y итерации по квадрату
    
    unsigned blocksize = 1;
    unsigned blocknum = polygons->size();

    matrix* d_m = transferMatrixToDevice(img);

    vertex* d_vertices;
    polygon* d_polygons;
    long long* d_zbuffer;
    unsigned char* d_polyColors;

    cuda_log(cudaMalloc(&d_vertices, vertices->size() * sizeof(vertex)));
    cuda_log(cudaMalloc(&d_polygons, polygons->size() * sizeof(polygon)));
    cuda_log(cudaMalloc(&d_polyColors, polygons->size() * sizeof(unsigned char)));
    cudaMalloc(&d_zbuffer, img->width * img->height * sizeof(long long));

    cuda_log(cudaMemcpy(d_vertices, vertices->data(), vertices->size() * sizeof(vertex), cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_polygons, polygons->data(), polygons->size() * sizeof(polygon), cudaMemcpyHostToDevice));


    kernel_fill<<<img->width * img->height, 1>>>(d_zbuffer, std::numeric_limits<long long>().max());

    //by default the limit on the number of simultaneous kernel launches is imposed 
    cuda_log(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, blocknum));

    kernel_drawPolygonsFilled<<<blocknum, blocksize>>>(d_m, d_vertices, d_polygons, vertices->size(), polygons->size(), scale, offset, randState, time(NULL), d_zbuffer, d_polyColors);
    cuda_log(cudaDeviceSynchronize());

    long long* h_zbuffer = new long long[img->width * img->height];
    unsigned char* h_polyColors = new unsigned char[polygons->size()];
    cuda_log(cudaMemcpy(h_polyColors, d_polyColors, polygons->size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cuda_log(cudaMemcpy(h_zbuffer, d_zbuffer, img->width * img->height * sizeof(long long), cudaMemcpyDeviceToHost));

    // for (size_t i = 0; i < polygons->size(); i++)
    // {
    //     vertex poly_v1(
    //         (*vertices)[(*polygons)[i].vertex_index1-1].x,
    //         (*vertices)[(*polygons)[i].vertex_index1-1].y,
    //         (*vertices)[(*polygons)[i].vertex_index1-1].z
    //     );
    //     vertex poly_v2(
    //         (*vertices)[(*polygons)[i].vertex_index2-1].x,
    //         (*vertices)[(*polygons)[i].vertex_index2-1].y,
    //         (*vertices)[(*polygons)[i].vertex_index2-1].z
    //     );
    //     vertex poly_v3{
    //         (*vertices)[(*polygons)[i].vertex_index3-1].x,
    //         (*vertices)[(*polygons)[i].vertex_index3-1].y,
    //         (*vertices)[(*polygons)[i].vertex_index3-1].z
    //     };

    //     //retrieve polygon's vertices and scale them
    //     vertex v1{
    //         scale * poly_v1.x + offset,
    //         img->height - (scale * poly_v1.y + offset),
    //         poly_v1.z
    //     };
    //     vertex v2{
    //         scale * poly_v2.x + offset,
    //         img->height - (scale * poly_v2.y + offset),
    //         poly_v2.z
    //     };
    //     vertex v3{
    //         scale * poly_v3.x + offset,
    //         img->height - (scale * poly_v3.y + offset),
    //         poly_v3.z
    //     };

    // }
    
    printf("____\t");
    for (unsigned i = 0; i < img->width; i++)
    {
        printf("%d\t", i+1);
    }

    printf("\n");
    for (unsigned i = 0; i < img->height; i++)
    {
        printf("%d\t", i+1);
        for (size_t j = 0; j < img->width; j++)
        {
            printf("%lld\t", h_zbuffer[img->width * i + j]);
        }
        printf("\n");
    }
    

    transferMatrixDataToHost(img, d_m);
}

#include "_image_draw_objects_instances.h"
