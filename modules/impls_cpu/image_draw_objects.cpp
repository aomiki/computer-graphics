#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include "vertex_tools.h"
#include "image_draw_objects.h"

template<typename E>
void draw_vertices(matrix_color<E>* m, vertices* verts, E vertex_color, float scaleX, float scaleY)
{
    for (size_t i = 0; i < verts->size; i++)
    {
        matrix_coord img_center((unsigned)(m->width/2), (unsigned)(m->height/2));

        int x = static_cast<int>(scaleX * verts->x[i] / verts->z[i] + img_center.x);
        int y = static_cast<int>(m->height - (scaleY * verts->y[i] / verts->z[i]  + img_center.y));

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

template <typename E>
void draw_polygons_filled(matrix_color<E>* img, vertices* verts, polygons* polys, float scaleX, float scaleY, unsigned char* modelColor)
{
    std::vector<float> zbuffer(img->width * img->height, std::numeric_limits<float>::max());
    for (size_t i = 0; i < polys->size; i++)
    {
        vertex poly_v1(
            verts->x[polys->vertex_index1[i]-1],
            verts->y[polys->vertex_index1[i]-1],
            verts->z[polys->vertex_index1[i]-1]
        );
        vertex poly_v2(
            verts->x[polys->vertex_index2[i]-1],
            verts->y[polys->vertex_index2[i]-1],
            verts->z[polys->vertex_index2[i]-1]
        );
        vertex poly_v3(
            verts->x[polys->vertex_index3[i]-1],
            verts->y[polys->vertex_index3[i]-1],
            verts->z[polys->vertex_index3[i]-1]
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

#include "_image_draw_objects_instances.h"