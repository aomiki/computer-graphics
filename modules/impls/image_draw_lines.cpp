#include "image_draw_lines.h"
#include <math.h>

void draw_star(matrix_rgb* matrix, matrix_coord from, color_rgb line_color, void (&draw_line)(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color))
{
    for (size_t i = 0; i < 13; i++)
    {
        double alpha = 2*M_PI* i/ 13;

        matrix_coord to(100+95*cos(alpha), 100+95*sin(alpha));
    
        draw_line(matrix, from, to, line_color);
    }
}

void draw_line_interpolation_count(matrix_rgb* matrix, matrix_coord from, matrix_coord to, int count, color_rgb line_color)
{
    double step = 1.0 / count;
    
    for (double i = 0; i < 1; i += step)
    {
        unsigned x = round((1.0 - i)*from.x + i*to.x);
        unsigned y = round((1.0 - i)*from.y + i*to.y);
        matrix->set(x,y, line_color);
    }
}