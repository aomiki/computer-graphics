#include "vertex_tools.h"
#include <cblas.h>
#include <math.h>


vertex_transforms::vertex_transforms()
{
}

vertex_transforms::~vertex_transforms()
{
}

void vertex_transforms::rotateAndOffset(vertex* vertices_transformed, vertex* vertices, unsigned n_vert, float offsets[3], float angles[3])
{
    float cosx = cos(angles[0]), sinx = sin(angles[0]);
    float cosy = cos(angles[1]), siny = sin(angles[1]);
    float cosz = cos(angles[2]), sinz = sin(angles[2]);

    const float rot_x[9] = {
        1, 0, 0,
        0, cosx, sinx,
        0, -sinx, cosx
    };

    const float rot_y[9] = {
        cosy, 0, siny,
        0, 1, 0,
        -siny, 0, cosy
    };

    const float rot_z[9] = {
        cosz,  sinz, 0,
        -sinz, cosz, 0,
        0, 0, 1
    };

    float rot_xy[9];
    float rot_xyz[9];
    float result[3];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, rot_x, 3, rot_y, 3, 0.0, rot_xy, 3);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, rot_xy, 3, rot_z, 3, 0.0, rot_xyz, 3);

    for (unsigned i = 0; i < n_vert; i++)
    {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, rot_xyz, 3, vertices[i].array, 1, 0.0, result, 1);
        vertices_transformed[i].x = result[0] + offsets[0];
        vertices_transformed[i].y = result[1] + offsets[1];
        vertices_transformed[i].z = result[2] + offsets[2];
    }
}
