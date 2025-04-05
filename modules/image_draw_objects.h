#ifndef IMAGE_DRAW_OBJECTS_H
#define IMAGE_DRAW_OBJECTS_H
#include "image_tools.h"
#include "obj_parser.h"

template<typename E>
void draw_vertices(matrix_color<E>* m, std::vector<vertex>* vertices, E vertex_color, int scale, int offset);

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons(matrix_color<E>* img, std::vector<vertex>* vertices, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons_filled(matrix_color<E> *img, std::vector<vertex> *vertices, std::vector<polygon> *polygons, float scaleX, float scaleY, unsigned char* modelColor);

__shared_func__ void calc_triangle_boundaries(matrix_coord& min_coord, matrix_coord& max_coord, vertex& v1, vertex& v2, vertex& v3, matrix& m);

#endif
