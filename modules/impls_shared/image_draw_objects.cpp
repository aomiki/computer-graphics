#include "image_draw_objects.h"
#include <cmath>

__shared_func__ void calc_triangle_boundaries(matrix_coord& min_coord, matrix_coord& max_coord, vertex& v1, vertex& v2, vertex& v3, matrix& m)
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
