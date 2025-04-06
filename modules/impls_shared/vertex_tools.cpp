#include "vertex_tools.h"
#include "image_draw_objects.h"
#include "image_tools.h"

#ifdef __CUDACC__
#define sqrt sqrtf
#else
#include <cmath>
#endif

__shared_func__ void get_barycentric_coords(vec3& baryc, screen_coords& m_coords, vec3 &v1, vec3 &v2, vec3 &v3)
{
    baryc.x = ((m_coords.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (m_coords.y - v3.y)) /
        ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y));

    baryc.y = ((v1.x - v3.x) * (m_coords.y - v3.y) - (m_coords.x - v3.x) * (v1.y - v3.y)) /
        ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y));

    baryc.z = 1.0 - baryc.x - baryc.y;
}

__shared_func__ float length(vec3 &vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

__shared_func__ float dot(vec3 &vec1, vec3 &vec2)
{
    return
        vec1.x * vec2.x +
        vec1.y * vec2.y +
        vec1.z * vec2.z;
}

__shared_func__ void normal(vec3& result, vec3 &vec1, vec3 &vec2)
{
    //https://en.wikipedia.org/wiki/Cross_product

    result = {
        vec1.y * vec2.z - vec1.z * vec2.y,
        vec1.z * vec2.x - vec1.x * vec2.z,
        vec1.x * vec2.y - vec1.y * vec2.x
    };
}

__shared_func__ void poly_vertices_to_vectors(
    vec3& poly_v1, vec3& poly_v2, vec3& poly_v3,
    vec3& vec1, vec3& vec2)
{
    vec1.x = poly_v2.x - poly_v3.x;
    vec1.y = poly_v2.y - poly_v3.y;
    vec1.z = poly_v2.z - poly_v3.z;

    vec2.x = poly_v2.x - poly_v1.x;
    vec2.y = poly_v2.y - poly_v1.y;
    vec2.z = poly_v2.z - poly_v1.z;
}
