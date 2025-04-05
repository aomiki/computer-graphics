#include "image_draw_objects.h"
#include "image_tools.h"

#ifdef __CUDACC__
#define sqrt sqrtf
#else
#include <cmath>
#include "vertex_tools.h"
#endif

__shared_func__ vertex get_barycentric_coords(matrix_coord &m_coords, vertex &v1, vertex &v2, vertex &v3)
{
    vertex baryc;

    baryc.x = ((m_coords.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (m_coords.y - v3.y)) /
        ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y));

    baryc.y = ((v1.x - v3.x) * (m_coords.y - v3.y) - (m_coords.x - v3.x) * (v1.y - v3.y)) /
        ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y));

    baryc.z = 1.0 - baryc.x - baryc.y;

    return baryc;
}

__shared_func__ double length(vertex &vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

__shared_func__ double dot(vertex &vec1, vertex &vec2)
{
    return
        vec1.x * vec2.x +
        vec1.y * vec2.y +
        vec1.z * vec2.z;
}

__shared_func__ vertex normal(vertex &vec1, vertex &vec2)
{
    //https://en.wikipedia.org/wiki/Cross_product

    vertex normal_vec(
        vec1.y * vec2.z - vec1.z * vec2.y,
        vec1.z * vec2.x - vec1.x * vec2.z,
        vec1.x * vec2.y - vec1.y * vec2.x
    );

    return normal_vec;
}

__shared_func__ void poly_vertices_to_vectors(
    vertex& poly_v1, vertex& poly_v2, vertex& poly_v3,
    vertex& vec1, vertex& vec2)
{
    vec1.x = poly_v2.x - poly_v3.x;
    vec1.y = poly_v2.y - poly_v3.y;
    vec1.z = poly_v2.z - poly_v3.z;

    vec2.x = poly_v2.x - poly_v1.x;
    vec2.y = poly_v2.y - poly_v1.y;
    vec2.z = poly_v2.z - poly_v1.z;
}