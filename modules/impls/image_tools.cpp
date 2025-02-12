#include "image_tools.h"

void matrix_gray::fill(char value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            array.push_back(value);
        }
    }
}

void matrix_gray::set(unsigned x, unsigned y, char color)
{
    size_t index = width*y+x;
    array[index] = color;
}

void matrix_rgb::fill(color_rgb value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            array.push_back(value.red);
            array.push_back(value.green);
            array.push_back(value.blue);
        }
    }
}

void matrix_rgb::set(unsigned x, unsigned y, color_rgb color)
{
    size_t index = (width*y+x)*3;

    (array)[index] = color.red;
    (array)[index+1] = color.green;
    (array)[index+2] = color.blue;
};