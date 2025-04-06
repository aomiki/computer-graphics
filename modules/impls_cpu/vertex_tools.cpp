#include "vertex_tools.h"
#include <cblas.h>
#include <math.h>


vertex_transforms::vertex_transforms()
{
}

vertex_transforms::~vertex_transforms()
{
}

void vertex_transforms::rotateAndOffset(vertices* verts_transformed, vertices* verts, float offsets[3], float angles[3])
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

    float* verts_membuf = new float[verts->size * 3];
    verts_transformed->x = verts_membuf;
    verts_transformed->y = verts_membuf + verts->size;
    verts_transformed->z = verts_membuf + verts->size * 2;

    for (unsigned i = 0; i < verts->size; i++)
    {
        float vert[3] { verts->x[i], verts->y[i], verts->z[i] };

        cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, rot_xyz, 3, vert, 1, 0.0, result, 1);
        verts_transformed->x[i] = result[0] + offsets[0];
        verts_transformed->y[i] = result[1] + offsets[1];
        verts_transformed->z[i] = result[2] + offsets[2];
    }
}
