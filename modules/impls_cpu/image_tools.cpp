#include "image_tools.h"
#include <cstring>

matrix::matrix(unsigned int components_num, unsigned width, unsigned height)
{
    this->height = 0;
    this->width = 0;
    this->components_num = components_num;
    set_arr_interlaced(nullptr);

    resize(width, height);
}

void matrix::resize(unsigned width, unsigned height)
{
    unsigned int old_size = size_interlaced();

    this->width = width;
    this->height = height;

    unsigned char* newArr = new unsigned char[size_interlaced()];

    if (old_size != 0)
    {
        delete [] arr;
    }

    set_arr_interlaced(newArr);
}

void matrix::fill(unsigned char *value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            unsigned char* cell = get(i, j);
            for (size_t k = 0; k < components_num; k++)
            {
                cell[k] = value[k];
            }
        }
    }
}

matrix::~matrix()
{
    if (size_interlaced() != 0 && arr != nullptr)
    {
        delete [] arr;
    }
}
