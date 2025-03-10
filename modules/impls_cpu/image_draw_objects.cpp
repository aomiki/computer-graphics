#include <cmath>
#include <algorithm>
#include <random>
#include "image_draw_objects.h"

template<typename E>
void draw_vertices(matrix_color<E>* m, std::vector<vertex>* vertices, E vertex_color, int scale, int offset)
{
    for (size_t i = 0; i < vertices->size(); i++)
    {
        int x = static_cast<int>(scale * (*vertices)[i].x + offset);
        int y = static_cast<int>(m->height - (scale * (*vertices)[i].y + offset));
        m->set(x, y, vertex_color);
    }
};

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3)
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
    else if(xmax > img->width)
    {
        xmax = img->width;
    }

    if (ymin < 0)
    {
        ymin = 0;
    }
    else if (ymax > img->height)
    {
        ymax = img->height;
    }

    unsigned xmin_i = round(xmin);
    unsigned xmax_i = round(xmax);

    unsigned ymin_i = round(ymin);
    unsigned ymax_i = round(ymax);

    matrix_coord curr(0, 0);
    for (curr.x = xmin_i; curr.x < xmax_i; curr.x++)
    {
        for (curr.y = ymin_i; curr.y < ymax_i; curr.y++)
        {
            vertex baryc = get_barycentric_coords(curr, v1, v2, v3);

            if ((baryc.x >= 0) && (baryc.y >= 0) && (baryc.z >= 0))
            {
                img->set(curr, polyg_color);
            }
        }
    }
};

template <typename E>
inline void draw_polygons_filled(matrix_color<E> *img, std::vector<vertex> *vertices, std::vector<polygon> *polygons, int scale, int offset)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned char> dist(0, 255);


    for (size_t i = 0; i < polygons->size(); i++)
    {
        unsigned char* c_polyg_color = new unsigned char[img->components_num];
        for (unsigned i = 0; i < img->components_num; i++)
        {
            c_polyg_color[i] = dist(mt);
        }
    
        E polyg_color = img->c_arr_to_element(c_polyg_color);

        vertex v1{
            scale * (*vertices)[(*polygons)[i].vertex_index1-1].x + offset,
            img->height - (scale * (*vertices)[(*polygons)[i].vertex_index1-1].y + offset)
        };
        vertex v2{
            scale * (*vertices)[(*polygons)[i].vertex_index2-1].x + offset,
            img->height - (scale * (*vertices)[(*polygons)[i].vertex_index2-1].y + offset)
        };
        vertex v3{
            scale * (*vertices)[(*polygons)[i].vertex_index3-1].x + offset,
            img->height - (scale * (*vertices)[(*polygons)[i].vertex_index3-1].y + offset)
        };

        draw_polygon(img, polyg_color, v1, v2, v3);

        delete [] c_polyg_color;
    }

}

#include "_image_draw_objects_instances.h"
