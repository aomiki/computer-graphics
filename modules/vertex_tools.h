#ifndef VERTEX_TOOLS_H
#define VERTEX_TOOLS_H

#include "image_tools.h"
#include "obj_parser.h"


#ifdef __CUDACC__
    #define vec3  float3
    #define screen_coords uint2
#else
    #define vec3 vertex
    #define screen_coords matrix_coord
#endif

__shared_func__ void get_barycentric_coords(vec3& result, screen_coords& m_coords, vec3& v1, vec3& v2, vec3& v3);
__shared_func__ float length(vec3& vec);
__shared_func__ float dot(vec3& vec1, vec3& vec2);
__shared_func__ void normal(vec3& result, vec3& vec1, vec3& vec2);

__shared_func__ void poly_vertices_to_vectors(
    vec3& poly_v1, vec3& poly_v2, vec3& poly_v3,
    vec3& vec1, vec3& vec2);


class vertex_transforms
{
    public:
        vertex_transforms();
        ~vertex_transforms();

        void rotateAndOffset(vertices* verts_transformed, vertices* verts, unsigned n_verts, float offsets[3], float angles[3]);
};

#endif
