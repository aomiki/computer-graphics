#include "include/lodepng.h"
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

std::string source_file_path = "/home/aomiki/Pictures/2by2_16bits.png";
std::string result_file_path = "result.png";

struct color_rgb {
    unsigned char red = 0;
    unsigned char green = 0;
    unsigned char blue = 0;
};

struct matrix_rgb {
    std::vector<unsigned char> array;
    unsigned width;
    unsigned height;


    void set(unsigned x, unsigned y, color_rgb color)
    {
        size_t index = (width*y+x)*3;

        (array)[index] = color.red;
        (array)[index+1] = color.green;
        (array)[index+2] = color.blue;
    }
};

void lr1_task1_img_black(unsigned width, unsigned height);
void lr1_task1_img_white(unsigned width, unsigned height);
void lr1_task1_img_red(unsigned width, unsigned height);
void lr1_task1_img_gradient(unsigned width, unsigned height);
void encode(std::vector<unsigned char>* img_source, std::vector<unsigned char>* img_buffer, unsigned width, unsigned height, LodePNGColorType colortype, unsigned bit_depth);
void img_fill_rgb(std::vector<unsigned char>* img, unsigned width, unsigned height, char val_r, char val_g, char val_b);
void save_png(std::vector<unsigned char>* png_buffer);
void dotted_line(matrix_rgb* matrix, unsigned x0, unsigned y0, unsigned x1, unsigned y1, int count, color_rgb line_color);
void lr1_task2_line(void (&draw_line)(matrix_rgb *matrix, unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1, int count, color_rgb line_color));

int main()
{
    std::cout << "Shellow from SSAU!" << std::endl;

    result_file_path = "t1_img_black.png";
    lr1_task1_img_black(100, 100);

    result_file_path = "t1_img_white.png";
    lr1_task1_img_white(100, 100);

    result_file_path = "t1_img_red.png";
    lr1_task1_img_red(100, 100);

    result_file_path = "t1_img_gradient.png";
    lr1_task1_img_gradient(255, 255);

    result_file_path = "t2_img_lines17.png";

    lr1_task2_line(dotted_line);

    std::cout << "that's it" << std::endl;
}

void lr1_task2_line(void (&draw_line)(matrix_rgb *matrix, unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1, int count, color_rgb line_color))
{
    std::vector<unsigned char> png;
    matrix_rgb matrix;
    matrix.width = 200;
    matrix.height = 200;
    
    img_fill_rgb(&matrix.array, matrix.width, matrix.height, 95, 164, 237);
    color_rgb line_color;
    line_color.red = 249;
    line_color.green = 89;
    line_color.blue = 255;

    for (size_t i = 0; i < 13; i++)
    {
        double alpha = 2*M_PI* i/ 13;

        unsigned x1 = 100+95*cos(alpha), y1 = 100+95*sin(alpha);
    
        draw_line(&matrix, 100, 100, x1, y1, 1000, line_color);
    }

    encode(&matrix.array, &png, matrix.width, matrix.height, LodePNGColorType::LCT_RGB, 8);
    save_png(&png);
}

void dotted_line(matrix_rgb* matrix, unsigned x0, unsigned y0, unsigned x1, unsigned y1, int count, color_rgb line_color)
{
    double step = 1.0 / count;
    
    for (double i = 0; i < 1; i += step)
    {
        unsigned x = round((1.0 - i)*x0 + i*x1);
        unsigned y = round((1.0 - i)*y0 + i*y1);
        matrix->set(x,y, line_color);
    }
}

void img_fill(std::vector<unsigned char>* img, unsigned width, unsigned height, char value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            img->push_back(value);
        }
    }
}

void img_fill_gradient(std::vector<unsigned char>* img, unsigned width, unsigned height, char val_r, char val_b)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            img->push_back(val_r);
            img->push_back(j);
            img->push_back(val_b);
        }
    }
}

void img_fill_rgb(std::vector<unsigned char>* img, unsigned width, unsigned height, char val_r, char val_g, char val_b)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            img->push_back(val_r);
            img->push_back(val_g);
            img->push_back(val_b);
        }
    }
}

void encode(std::vector<unsigned char>* img_source, std::vector<unsigned char>* img_buffer, unsigned width, unsigned height, LodePNGColorType colortype, unsigned bit_depth)
{
    lodepng::encode(*img_buffer, *img_source, width, height, colortype, bit_depth);
}

void decode(std::vector<unsigned char>* img_source, std::vector<unsigned char>* img_buffer, unsigned width, unsigned height, LodePNGColorType colortype, unsigned bit_depth)
{
   lodepng::decode(*img_buffer, width, height, *img_source, colortype, bit_depth);
}

void load_png(std::vector<unsigned char>* png_buffer)
{
    lodepng::load_file(*png_buffer, source_file_path);
}

void save_png(std::vector<unsigned char>* png_buffer)
{
    lodepng::save_file(*png_buffer, result_file_path);
}

void lr1_task1_img_black(unsigned width, unsigned height)
{
    std::vector<unsigned char> img;
    std::vector<unsigned char> png_buffer;

    img_fill(&img, width, height, 0);
    encode(&img, &png_buffer, width, height, LodePNGColorType::LCT_GREY, 8);
    save_png(&png_buffer);
}

void lr1_task1_img_white(unsigned width, unsigned height)
{
    std::vector<unsigned char> img;
    std::vector<unsigned char> png_buffer;

    img_fill(&img, width, height, 255);
    encode(&img, &png_buffer, width, height, LodePNGColorType::LCT_GREY, 8);
    save_png(&png_buffer);
}

void lr1_task1_img_red(unsigned width, unsigned height)
{
    std::vector<unsigned char> img;
    std::vector<unsigned char> png_buffer;

    img_fill_rgb(&img, width, height, 255, 0, 0);
    encode(&img, &png_buffer, width, height, LodePNGColorType::LCT_RGB, 8);
    save_png(&png_buffer);
}

void lr1_task1_img_gradient(unsigned width, unsigned height)
{
    std::vector<unsigned char> img;
    std::vector<unsigned char> png_buffer;

    img_fill_gradient(&img, width, height, 95, 237);
    encode(&img, &png_buffer, width, height, LodePNGColorType::LCT_RGB, 8);
    save_png(&png_buffer);
}
