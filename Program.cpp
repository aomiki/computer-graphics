#include "lodepng.h"
#include "image_codec.h"
#include "LR1.h"
#include "image_draw_lines.h"

#include <iostream>
#include <vector>
#include <string>

std::string result_folder = "output";
std::string input_folder = "input";
std::string lr1_result_folder = result_folder + "/LR1/";

void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void decode_encode_img(std::string filepath, image_codec* codec);

int main()
{
    std::cout << "Shellow from SSAU!" << std::endl;

    image_codec codec;

    decode_encode_img("shuttle.jpg", &codec);

    lr1_task1_img_black(4000, 2000, lr1_result_folder + "t1_img_black", &codec);

    lr1_task1_img_white(10, 10, lr1_result_folder + "t1_img_white", &codec);

    lr1_task1_img_red(4000, 2000, lr1_result_folder + "t1_img_red", &codec);

    lr1_task1_img_gradient(255, 255, lr1_result_folder + "t1_img_gradient", &codec);
    
    lr1_task2_line(dotted_line_count, lr1_result_folder + "t2_img_line_interp_count", &codec);
    lr1_task2_line(draw_line_interpolation, lr1_result_folder + "t2_img_line_interp_autocount", &codec);
    lr1_task2_line(draw_line_interpolation_xloop, lr1_result_folder + "t2_img_line_interp_xloop", &codec);
    lr1_task2_line(draw_line_interpolation_xloop_fixX, lr1_result_folder + "t2_img_line_interp_xloop_fixX", &codec);
    lr1_task2_line(draw_line_interpolation_xloop_fixXfixY, lr1_result_folder + "t2_img_line_interp_xloop_fixXfixY", &codec);
    lr1_task2_line(draw_line_dy, lr1_result_folder + "t2_img_line_dy", &codec);
    lr1_task2_line(draw_line_dy_rev1, lr1_result_folder + "t2_img_line_dy_rev1", &codec);
    lr1_task2_line(draw_line, lr1_result_folder + "t2_img_line_bresenham", &codec);
    codec.~image_codec();

    std::cout << "that's it" << std::endl;
}

void decode_encode_img(std::string filepath, image_codec* codec)
{
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder+ "/" +filepath);

    matrix_rgb img_matrix;
    codec->decode(&img_buffer, &img_matrix,ImageColorScheme::IMAGE_RGB, 8);

    unsigned int vert_boundary = (int)img_matrix.height/10;
    unsigned int horiz_boundary = (int)img_matrix.width/10;

    for (size_t i = 0; i < img_matrix.height; i++)
    {
        for (size_t j = 0; j < img_matrix.width; j++)
        {
            if (i < vert_boundary || i > img_matrix.height - vert_boundary)
            {
                img_matrix.set(j, i, color_rgb(255, 255, 255));
            }
            else if (j < horiz_boundary || j > img_matrix.width - horiz_boundary)
            {
                img_matrix.set(j, i, color_rgb(255, 255, 255));   
            }
        }
    }

    img_buffer.clear();
    codec->encode(&img_buffer, &img_matrix, ImageColorScheme::IMAGE_RGB, 8);

    codec->save_image_file(&img_buffer, result_folder+"/"+filepath);
}

void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    unsigned count = 1000;
    draw_line_interpolation_count(matrix, from, to, count, line_color);
}

