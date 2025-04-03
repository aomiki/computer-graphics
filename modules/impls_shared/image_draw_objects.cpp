#include "image_draw_objects.h"
#include <cmath>

__shared_func__ void calc_triangle_boundaries(matrix_coord& min_coord, matrix_coord& max_coord, vertex& screen_v1, vertex& screen_v2, vertex& screen_v3, matrix& m)
{
    double scr_xmin = min(min(screen_v1.x, screen_v2.x), screen_v3.x);
    double scr_ymin = min(min(screen_v1.y, screen_v2.y), screen_v3.y);

    double scr_xmax = max(max(screen_v1.x, screen_v2.x), screen_v3.x)+1;
    double scr_ymax = max(max(screen_v1.y, screen_v2.y), screen_v3.y)+1;

    //crop to img boundaries
    if (scr_xmin < 0)
    {
        scr_xmin = 0;
    }
    
    if(scr_xmax > m.width)
    {
        scr_xmax = m.width;
    }

    if (scr_ymin < 0)
    {
        scr_ymin = 0;
    }
    
    if (scr_ymax > m.height)
    {
        scr_ymax = m.height;
    }

    min_coord.x = round(scr_xmin);
    max_coord.x = round(scr_xmax);

    min_coord.y = round(scr_ymin);
    max_coord.y = round(scr_ymax);
};
