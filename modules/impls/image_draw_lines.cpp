#include "image_draw_lines.h"
#include <cmath>
#include <cstdint>

int square(int x)
{
    return x * x;
}

void draw_star(matrix_rgb* matrix, matrix_coord from, color_rgb line_color, void (&draw_line)(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color))
{
    for (size_t i = 0; i < 13; i++)
    {
        float alpha = 2*M_PI* i/ 13;

        matrix_coord to(100+95*cos(alpha), 100+95*sin(alpha));
    
        draw_line(matrix, from, to, line_color);
    }
}

void draw_line_interpolation_count(matrix_rgb* matrix, matrix_coord from, matrix_coord to, int count, color_rgb line_color)
{
    float step = 1.0 / count;
    
    for (float i = 0; i < 1; i += step)
    {
        unsigned x = round((1.0 - i)*from.x + i*to.x);
        unsigned y = round((1.0 - i)*from.y + i*to.y);
        matrix->set(x,y, line_color);
    }
}

void draw_line_interpolation(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    float count = sqrt(square(from.x-to.x) + square(from.y-to.y));
    float step = 1.0 / count;

    for (float i = 0; i < 1; i += step)
    {
        unsigned x = round((1.0 - i)*from.x + i*to.x);
        unsigned y = round((1.0 - i)*from.y + i*to.y);
        matrix->set(x,y, line_color);
    }
}

void draw_line_interpolation_xloop(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    for (float x = from.x; x < to.x; x++)
    {
        float t = (x-from.x)/(to.x - from.x);

        unsigned y = round((1.0 - t)*from.y + t*to.y);
        matrix->set(x,y, line_color);
    }
}

void draw_line_interpolation_xloop_fixX(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    if (to.x < from.x)
    {
        std::swap(from, to);
    }
    
    for (float x = from.x; x < to.x; x++)
    {
        float t = (x-from.x)/(to.x - from.x);

        unsigned y = round((1.0 - t)*from.y + t*to.y);
        matrix->set(x,y, line_color);
    }
}

void draw_line_interpolation_xloop_fixXfixY(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{

    bool xchange = false;

    if (abs(from.x - to.x) < abs(from.y - to.y))
    {
        std::swap(to.x, to.y);
        std::swap(from.x, from.y);
        xchange = true;
    }

    if (to.x < from.x)
    {
        std::swap(from, to);
    }

    for (float x = from.x; x < to.x; x++)
    {
        float t = (x-from.x)/(to.x - from.x);

        unsigned y = round((1.0 - t)*from.y + t*to.y);

        if (xchange)
        {
            matrix->set(y,x, line_color);
        }
        else
        {
            matrix->set(x,y, line_color);
        }
    }
}

void draw_line_dy(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{

    bool xchange = false;

    if (abs(from.x - to.x) < abs(from.y - to.y))
    {
        std::swap(to.x, to.y);
        std::swap(from.x, from.y);
        xchange = true;
    }

    if (to.x < from.x)
    {
        std::swap(from, to);
    }

    float dy = ((float)abs(to.y - from.y))/(to.x - from.x);
    unsigned y = from.y;
    
    float derror = 0.0;

    char y_upd = to.y > from.y? 1 : -1;

    for (unsigned x = from.x; x < to.x; x++)
    {
        if (xchange)
        {
            matrix->set(y,x, line_color);
        }
        else
        {
            matrix->set(x,y, line_color);
        }

        derror += dy;

        if (derror > 0.5)
        {
            derror -= 1.0;
            y+= y_upd;
        }
    }
}

void draw_line_dy_rev1(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{

    bool xchange = false;

    if (abs(from.x - to.x) < abs(from.y - to.y))
    {
        std::swap(to.x, to.y);
        std::swap(from.x, from.y);
        xchange = true;
    }

    if (to.x < from.x)
    {
        std::swap(from, to);
    }

    float dy = 2.0*(to.x - from.x)*((float)abs(to.y - from.y))/(to.x - from.x);
    unsigned y = from.y;
    
    float derror = 0.0;

    char y_upd = to.y > from.y? 1 : -1;

    for (unsigned x = from.x; x < to.x; x++)
    {
        if (xchange)
        {
            matrix->set(y,x, line_color);
        }
        else
        {
            matrix->set(x,y, line_color);
        }

        derror += dy;

        if (derror > 2.0*(to.x - from.x)*0.5)
        {
            derror -= 2.0*(to.x - from.x)*1.0;
            y+= y_upd;
        }
    }
}

void draw_line(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    bool xchange = false;

    if (abs(from.x - to.x) < abs(from.y - to.y))
    {
        std::swap(to.x, to.y);
        std::swap(from.x, from.y);
        xchange = true;
    }

    if (to.x < from.x)
    {
        std::swap(from, to);
    }

    unsigned dy = 2*abs(to.y - from.y);
    unsigned y = from.y;
    
    long long derror = 0;

    char y_upd = to.y > from.y? 1 : -1;

    for (unsigned x = from.x; x < to.x; x++)
    {
        if (xchange)
        {
            matrix->set(y,x, line_color);
        }
        else
        {
            matrix->set(x,y, line_color);
        }

        derror += dy;

        if (derror > (to.x - from.x))
        {
            derror -= 2*(to.x - from.x);
            y+= y_upd;
        }
    }
}
