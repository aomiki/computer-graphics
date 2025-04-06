#include <vector>
#include "image_tools.h"
#include "obj_parser.h"
#include "image_draw_objects.h"

template void draw_vertices<unsigned char>(matrix_color<unsigned char>* m, vertices* verts, unsigned char vertex_color, float scaleX, float scaleY);
template void draw_vertices<color_rgb>(matrix_color<color_rgb>* m, vertices* verts, color_rgb vertex_color, float scaleX, float scaleY);

template void draw_polygon<unsigned char>(matrix_color<unsigned char>* img, unsigned char polyg_color, vertex v1, vertex v2, vertex v3);
template void draw_polygon<color_rgb>(matrix_color<color_rgb>* img, color_rgb polyg_color, vertex v1, vertex v2, vertex v3);

template void draw_polygons_filled<unsigned char>(matrix_color<unsigned char> *img, vertices *verts, polygons* polys, float scaleX, float scaleY, unsigned char* modelColor);
template void draw_polygons_filled<color_rgb>(matrix_color<color_rgb> *img, vertices* verts, polygons* polys, float scaleX, float scaleY, unsigned char* modelColor);
