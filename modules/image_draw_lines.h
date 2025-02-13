#include "image_tools.h"

/// @brief Рисует линию используя линейную интерполяцию. Кол-во закрашеных точек равно count.
void draw_line_interpolation_count(matrix_rgb* matrix, matrix_coord from, matrix_coord to, int count, color_rgb line_color);

/// @brief Рисует линию используя линейную интерполяцию. Кол-во закрашеных точек вычисляется автоматически.
void draw_line_interpolation(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color);

void draw_line_interpolation_xloop(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void draw_line_interpolation_xloop_fixX(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void draw_line_interpolation_xloop_fixXfixY(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);

void draw_line_dy(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void draw_line_dy_rev1(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);

/// @brief Рисует линию используя алгоритм Брезенхама.
/// @param matrix 
/// @param from 
/// @param to 
/// @param line_color 
void draw_line(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);

/// @brief Рисует линии используя draw_line в виде звезды.
/// @param matrix 
/// @param from 
/// @param line_color 
/// @param draw_line 
void draw_star(matrix_rgb* matrix, matrix_coord from, color_rgb line_color, void (&draw_line)(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color));
