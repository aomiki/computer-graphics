#include "image_tools.h"
#include "obj_parser.h"

vertex get_barycentric_coords(matrix_coord m_coords, vertex triag_vert1, vertex triag_vert2, vertex triag_vert3);

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3);

#include "impls/image_draw_objects.ipp"
