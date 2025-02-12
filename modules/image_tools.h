#include <vector>

#ifndef image_tools_h
#define image_tools_h

struct color_rgb {
    color_rgb(unsigned char r, unsigned char g, unsigned char b)
    {
        red = r;
        green = g;
        blue = b;
    }

    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

struct matrix_coord {
    matrix_coord(unsigned x, unsigned y)
    {
        this->x = x;
        this->y = y;
    }

    unsigned x;
    unsigned y;
};

class matrix {
    public:
        std::vector<unsigned char> array;
        unsigned width;
        unsigned height;

        matrix(unsigned width, unsigned height);
};

template<typename E>
class matrix_color : public matrix {
    public:
    matrix_color(unsigned width, unsigned height) : matrix(width, height) {}
    void virtual set(unsigned x, unsigned y, E color) = 0;

    void virtual fill(E value);
};

class matrix_rgb : public matrix_color<color_rgb>
{
    public:
        matrix_rgb(unsigned width, unsigned height): matrix_color<color_rgb>(width, height) {}
        void virtual set(unsigned x, unsigned y, color_rgb color);
        void virtual fill(color_rgb value);
};

class matrix_gray : public matrix_color<char>
{
    public:
        matrix_gray(unsigned width, unsigned height): matrix_color<char>(width, height) {}
        void virtual set(unsigned x, unsigned y, char color);
        void virtual fill(char value);
};

#include "impls/image_tools.ipp"

#endif