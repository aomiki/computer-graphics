#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include "vertex_tools.h"
#include "image_draw_objects.h"

model_renderer::model_renderer(vertices *verts, polygons *polys)
{
    this->n_polys = polys->size;
    this->n_verts = verts->size;

    //copy vertices
    d_verts = new vertices();
    memcpy(d_verts, verts, sizeof(vertices));

    float* verts_arr = new float[n_verts * 3];
    d_verts->x = verts_arr;
    d_verts->y = verts_arr + n_verts;
    d_verts->z = verts_arr + n_verts * 2;

    memcpy(d_verts->x, verts->x, n_verts * sizeof(float) * 3);

    //copy polygons
    d_polys = new polygons();
    memcpy(d_polys, polys, sizeof(polygons));

    unsigned* polys_arr = new unsigned[n_polys * 3];
    d_polys->vertex_index1 = polys_arr;
    d_polys->vertex_index2 = polys_arr + n_polys;
    d_polys->vertex_index3 = polys_arr + n_polys*2;

    memcpy(d_polys->vertex_index1, polys->vertex_index1, n_polys * sizeof(unsigned) * 3);

    this->d_verts_transformed = d_verts;
}

model_renderer::~model_renderer()
{
    if (d_verts_transformed != d_verts)
    {
        delete [] d_verts_transformed->x;
        delete d_verts_transformed;
    }

    delete [] d_verts->x;
    delete d_verts;

    delete [] d_polys->vertex_index1;
    delete d_polys;
}

template<typename E>
void model_renderer::draw_vertices(matrix_color<E>* m, E vertex_color, float scaleX, float scaleY)
{
    for (size_t i = 0; i < n_verts; i++)
    {
        matrix_coord img_center((unsigned)(m->width/2), (unsigned)(m->height/2));

        int x = static_cast<int>(scaleX * d_verts_transformed->x[i] / d_verts_transformed->z[i] + img_center.x);
        int y = static_cast<int>(m->height - (scaleY * d_verts_transformed->y[i] / d_verts_transformed->z[i]  + img_center.y));

        if (x >= m->width || y >= m->height)
            return;

        m->set(x, y, vertex_color);
    }
}

void drawPolygonInternal(matrix* img, matrix_coord min, matrix_coord max, unsigned char* polygon_color, vertex v1, vertex v2, vertex v3, std::vector<float>* zbuffer = nullptr)
{
    matrix_coord curr(0, 0);
    for (curr.x = min.x; curr.x < max.x; curr.x++)
    {
        for (curr.y = min.y; curr.y < max.y; curr.y++)
        {
            vertex baryc;
            get_barycentric_coords(baryc, curr, v1, v2, v3);
            if (baryc.x >= 0 && baryc.y >= 0 && baryc.z >= 0)
            {
                if (zbuffer == nullptr)
                {
                    std::memcpy(img->get(curr.x, curr.y), polygon_color, img->components_num);
                }
                else
                {
                    float interpolated_z = baryc.x * v1.z + baryc.y * v2.z + baryc.z * v3.z;
                    int buffer_index = curr.y * img->width + curr.x;
                    if (interpolated_z < (*zbuffer)[buffer_index])
                    {
                        (*zbuffer)[buffer_index] = interpolated_z;
                        std::memcpy(img->get(curr.x, curr.y), polygon_color, img->components_num); 
                    }
                }
                
            }
        }
    }
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
    unsigned char* vals = new unsigned char[img->components_num];
    img->element_to_c_arr(vals, polyg_color);
    drawPolygonInternal(img, min, max, vals, v1, v2, v3);
}


void model_renderer::rotateAndOffset(float offsets[3], float angles[3], vertex_transforms* vt_transformer)
{
    if (d_verts_transformed == d_verts)
    {
        //copy vertices
        d_verts_transformed = new vertices();
    
        float* verts_arr = new float[n_verts * 3];
        d_verts_transformed->x = verts_arr;
        d_verts_transformed->y = verts_arr + n_verts;
        d_verts_transformed->z = verts_arr + n_verts * 2;
    }

    vt_transformer->rotateAndOffset(d_verts_transformed, d_verts, n_verts, offsets, angles);
}

template<typename E>
void model_renderer::draw_polygons(matrix_color<E> *img, float scaleX, float scaleY, unsigned char* modelColor)
{
    std::vector<float> zbuffer(img->width * img->height, std::numeric_limits<float>::max());
    for (size_t i = 0; i < n_polys; i++)
    {
        vertex poly_v1(
            d_verts_transformed->x[d_polys->vertex_index1[i]-1],
            d_verts_transformed->y[d_polys->vertex_index1[i]-1],
            d_verts_transformed->z[d_polys->vertex_index1[i]-1]
        );
        vertex poly_v2(
            d_verts_transformed->x[d_polys->vertex_index2[i]-1],
            d_verts_transformed->y[d_polys->vertex_index2[i]-1],
            d_verts_transformed->z[d_polys->vertex_index2[i]-1]
        );
        vertex poly_v3(
            d_verts_transformed->x[d_polys->vertex_index3[i]-1],
            d_verts_transformed->y[d_polys->vertex_index3[i]-1],
            d_verts_transformed->z[d_polys->vertex_index3[i]-1]
        );

        unsigned char* c_polyg_color = new unsigned char[img->components_num];
        vertex poly_vec1, poly_vec2;
        poly_vertices_to_vectors(poly_v1, poly_v2, poly_v3, poly_vec1, poly_vec2);

        vertex normal_vec;
        normal(normal_vec, poly_vec1, poly_vec2);

        // Направление камеры (по оси Z)
        vertex camera_vec(0.0, 0.0, 1.0);
        float d = dot(normal_vec, camera_vec);
        float viewing_angle_cosine = d/(length(normal_vec)*length(camera_vec));

        if (viewing_angle_cosine >= 0)
        {
            delete[] c_polyg_color;
            continue;
        }

        c_polyg_color[0] = (unsigned char)(-255.0 * viewing_angle_cosine + 0.5);

        matrix_coord min(0,0);
        matrix_coord max(0,0);

        float u0 = img->width / 2.0; 
        float v0 = img->height / 2.0;
        vertex v1(
            (scaleX * poly_v1.x) / poly_v1.z + u0,
            img->height - ((scaleY * poly_v1.y ) / poly_v1.z + v0),
            poly_v1.z
        );
        vertex v2(
            (scaleX * poly_v2.x) / poly_v2.z + u0,
            img->height - ((scaleY * poly_v2.y) / poly_v2.z + v0),
            poly_v2.z
        );
        vertex v3(
            (scaleX * poly_v3.x) / poly_v3.z + u0,
            img->height - ((scaleY * poly_v3.y) / poly_v3.z + v0),
            poly_v3.z
        );

        calc_triangle_boundaries(min, max, v1, v2, v3, *img);
        if (max.x <= min.x || max.y <= min.y)
        {
            return;
        }

        drawPolygonInternal(img, min, max, c_polyg_color, v1, v2, v3, &zbuffer);
    }
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