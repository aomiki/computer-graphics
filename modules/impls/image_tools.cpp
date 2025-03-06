#include "image_tools.h"
#include "matrix_routines.h"

inline unsigned int matrix::get_interlaced_index(unsigned int x, unsigned int y)
{
    return (width*y+x)*COMPONENTS_NUM;
}

unsigned char* matrix::get(unsigned int x, unsigned int y)
{
    return array.data() + get_interlaced_index(x, y);
}

unsigned char* matrix::get_c_arr_interlaced()
{
    return array.data();
}

unsigned int matrix::size_interlaced()
{
    return array.size();
}


void matrix_gray::element_to_c_arr(unsigned char* buffer, unsigned char value)
{
    buffer[0] = value;
}

unsigned char matrix_gray::c_arr_to_element(unsigned char *buffer)
{
    return buffer[0];
}


void matrix_rgb::element_to_c_arr(unsigned char* buffer, color_rgb value)
{
    buffer[0] = value.red;
    buffer[1] = value.green;
    buffer[2] = value.blue;
}

color_rgb matrix_rgb::c_arr_to_element(unsigned char *buffer)
{
    return color_rgb(buffer[0], buffer[1], buffer[2]);
}
