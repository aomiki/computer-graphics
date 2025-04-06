#include "image_draw_objects.h"
#include "vertex_tools.h"
#include <cmath>

__shared_func__ void calc_triangle_boundaries(screen_coords& min_coord, screen_coords& max_coord, vec3& screen_v1, vec3& screen_v2, vec3& screen_v3, matrix& m)
{
    float scr_xmin = min(min(screen_v1.x, screen_v2.x), screen_v3.x);
    float scr_ymin = min(min(screen_v1.y, screen_v2.y), screen_v3.y);

    float scr_xmax = max(max(screen_v1.x, screen_v2.x), screen_v3.x)+1;
    float scr_ymax = max(max(screen_v1.y, screen_v2.y), screen_v3.y)+1;

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
