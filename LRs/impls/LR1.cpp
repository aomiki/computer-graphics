#include "LR1.h"
#include "image_codec.h"
#include "image_draw_lines.h"

void lr1_task2_line(void (&draw_line)(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color), std::string filepath)
{
    std::vector<unsigned char> png;
    matrix_rgb matrix(200, 200);
    color_rgb line_color(249, 89, 255);
    matrix_coord from(100, 100);

    matrix.fill(color_rgb(95, 164, 237));

    draw_star(&matrix, from, line_color, draw_line);

    encode(&png, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    save_image_file(&png, filepath);
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

void lr1_task1_img_black(unsigned width, unsigned height, std::string filepath)
{
    matrix_gray matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(0);

    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
    save_image_file(&png_buffer, filepath);
}

void lr1_task1_img_white(unsigned width, unsigned height, std::string filepath)
{
    matrix_gray matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(255);

    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
    save_image_file(&png_buffer, filepath);
}

void lr1_task1_img_red(unsigned width, unsigned height, std::string filepath)
{
    matrix_rgb matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(color_rgb(255, 0, 0));

    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    save_image_file(&png_buffer, filepath);
}

void lr1_task1_img_gradient(unsigned width, unsigned height, std::string filepath)
{
    matrix_rgb matrix(width, height);
    std::vector<unsigned char> png_buffer;

    img_fill_gradient(&matrix, color_rgb(95, 0, 237));
    encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    save_image_file(&png_buffer, filepath);
}
