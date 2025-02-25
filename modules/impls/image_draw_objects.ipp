#include <cmath>
#include <algorithm>

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
