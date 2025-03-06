#include "matrix_routines.h"

template<typename E>
void matrix_color<E>::fill(E value)
{
    array.resize(width * height * 3);

    #ifdef CUDA_IMPL
    unsigned char* c_value = new unsigned char[COMPONENTS_NUM];
    element_to_c_arr(c_value, value);

    fillInterlaced(this, c_value);

    delete [] c_value;

    #else

    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            set(i, j, value);
        }
    }

    #endif
}

template<typename E>
E matrix_color<E>::get(unsigned x, unsigned y)
{
    unsigned char* cell = matrix::get(x, y);
    E value = c_arr_to_element(cell);

    return value;
};

template<typename E>
void matrix_color<E>::set(matrix_coord coord, E value)
{
    set(coord.x, coord.y , value);
}

template<typename E>
void matrix_color<E>::set(unsigned x, unsigned y, E value)
{
    unsigned char* cell = matrix::get(x, y);
    element_to_c_arr(cell, value);
}

inline matrix::matrix(unsigned int components_num, unsigned width, unsigned height) : COMPONENTS_NUM(components_num)
{
    this->height = height;
    this->width = width;
}

inline matrix::matrix(unsigned int components_num) : COMPONENTS_NUM(components_num)
{
    this->height = 0;
    this->width = 0;
}