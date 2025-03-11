#ifndef IMAGE_DRAW_OBJECTS_H
#define IMAGE_DRAW_OBJECTS_H
#include "image_tools.h"
#include "obj_parser.h"

__shared_func__ vertex get_barycentric_coords(matrix_coord m_coords, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_vertices(matrix_color<E>* m, std::vector<vertex>* vertices, E vertex_color, int scale, int offset);

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons(matrix_color<E>* img, std::vector<vertex>* vertices, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons_filled(matrix_color<E> *img, std::vector<vertex> *vertices, std::vector<polygon> *polygons, int scale, int offset);

#endif
