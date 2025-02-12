#include "image_tools.h"

/// @brief Рисует линию используя линейную интерполяцию. Кол-во закрашеных точек равно count.
void draw_line_interpolation_count(matrix_rgb* matrix, matrix_coord from, matrix_coord to, int count, color_rgb line_color);

/// @brief Рисует линии используя draw_line в виде звезды.
/// @param matrix 
/// @param from 
/// @param line_color 
/// @param draw_line 
void draw_star(matrix_rgb* matrix, matrix_coord from, color_rgb line_color, void (&draw_line)(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color));