#include "image_draw_objects.h"
#include "image_tools.h"

vertex get_barycentric_coords(matrix_coord m_coords, vertex v1, vertex v2, vertex v3)
{
    vertex baryc;

    baryc.x = ((m_coords.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (m_coords.y - v3.y)) /
        ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y));

    baryc.y = ((v1.x - v3.x) * (m_coords.y - v3.y) - (m_coords.x - v3.x) * (v1.y - v3.y)) /
        ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y));

    baryc.z = 1.0 - baryc.x - baryc.y;

    return baryc;
}
