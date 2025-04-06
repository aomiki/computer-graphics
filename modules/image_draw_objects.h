#ifndef IMAGE_DRAW_OBJECTS_H
#define IMAGE_DRAW_OBJECTS_H
#include "vertex_tools.h"
#include "image_tools.h"
#include "obj_parser.h"

template<typename E>
void draw_vertices(matrix_color<E>* m, vertices* verts, E vertex_color, float scaleX, float scaleY);

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons(matrix_color<E>* img, vertices* verts, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons_filled(matrix_color<E> *img, vertices *verts, polygons *polys, float scaleX, float scaleY, unsigned char* modelColor);

__shared_func__ void calc_triangle_boundaries(screen_coords& min_coord, screen_coords& max_coord, vec3& screen_v1, vec3& screen_v2, vec3& screen_v3, matrix& m);

#endif
