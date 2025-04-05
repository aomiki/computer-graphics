#ifndef VERTEX_TOOLS_H
#define VERTEX_TOOLS_H

#include "image_tools.h"
#include "obj_parser.h"


__shared_func__ vertex get_barycentric_coords(matrix_coord& m_coords, vertex& v1, vertex& v2, vertex& v3);
__shared_func__ double length(vertex& vec);
__shared_func__ double dot(vertex& vec1, vertex& vec2);
__shared_func__ vertex normal(vertex& vec1, vertex& vec2);

__shared_func__ void poly_vertices_to_vectors(
    vertex& poly_v1, vertex& poly_v2, vertex& poly_v3,
    vertex& vec1, vertex& vec2);


class vertex_transforms
{
    public:
        vertex_transforms();
        ~vertex_transforms();

        void rotateAndOffset(vertex* vertices_transformed, vertex* vertices, unsigned n_vert, double offsets[3], double angles[3]);
};

#endif
