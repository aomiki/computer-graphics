#include "include/lodepng.h"
#include "modules/image_codec.h"

#include <iostream>
#include <vector>
#include <string>
#include <math.h>


std::string source_file_path = "/home/aomiki/Pictures/2by2_16bits.png";
std::string result_file_path = "result.png";

void lr1_task1_img_black(unsigned width, unsigned height);
void lr1_task1_img_white(unsigned width, unsigned height);
void lr1_task1_img_red(unsigned width, unsigned height);
void lr1_task1_img_gradient(unsigned width, unsigned height);
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
    matrix_rgb matrix(200, 200);

    matrix.fill(color_rgb(95, 164, 237));

    color_rgb line_color(249, 89, 255);

    for (size_t i = 0; i < 13; i++)
    {
        double alpha = 2*M_PI* i/ 13;

        unsigned x1 = 100+95*cos(alpha), y1 = 100+95*sin(alpha);
    
        draw_line(&matrix, 100, 100, x1, y1, 1000, line_color);
    }

    encode(&png, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    save_image_file(&png, result_file_path);
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

void img_fill_gradient(matrix_rgb* matrix, color_rgb basecolor)
{
    for (size_t i = 0; i < matrix->width; i++)
    {
        for (size_t j = 0; j < matrix->height; j++)
        {
            matrix->array.push_back(basecolor.red);
            matrix->array.push_back(j);
            matrix->array.push_back(basecolor.blue);
        }
    }
}

void lr1_task1_img_black(unsigned width, unsigned height)
{
    matrix_gray matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(0);

    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
    save_image_file(&png_buffer, result_file_path);
}

void lr1_task1_img_white(unsigned width, unsigned height)
{
    matrix_gray matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(255);

    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
    save_image_file(&png_buffer, result_file_path);
}

void lr1_task1_img_red(unsigned width, unsigned height)
{
    matrix_rgb matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(color_rgb(255, 0, 0));

    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    save_image_file(&png_buffer, result_file_path);
}

void lr1_task1_img_gradient(unsigned width, unsigned height)
{
    matrix_rgb matrix(width, height);
    std::vector<unsigned char> png_buffer;

    img_fill_gradient(&matrix, color_rgb(95, 0, 237));
    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    save_image_file(&png_buffer, result_file_path);
}
