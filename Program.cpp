#include "lodepng.h"
#include "image_codec.h"
#include "LR1.h"
#include "LR2.h"
#include "image_draw_lines.h"
#include "image_tools.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#define LR1

namespace fs = std::filesystem;

const fs::path result_folder("output");
const fs::path input_folder ("input");
const fs::path lr1_result_folder = result_folder / "LR1";
const fs::path lr2_result_folder = result_folder / "LR2";


void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void decode_encode_img(std::string filepath, image_codec* codec);

int main()
{
    std::cout << "Shellow from SSAU!" << std::endl;

    image_codec codec;

    #ifdef LR1
    decode_encode_img("shuttle.jpg", &codec);

    lr1_task1_img_black(4000, 2000, lr1_result_folder / "t1_img_black", &codec);

    lr1_task1_img_white(10, 10, lr1_result_folder / "t1_img_white", &codec);

    lr1_task1_img_red(4000, 2000, lr1_result_folder / "t1_img_red", &codec);

    lr1_task1_img_gradient(255, 255, lr1_result_folder / "t1_img_gradient", &codec);
    
    lr1_task2_line(dotted_line_count, lr1_result_folder / "t2_img_line_interp_count", &codec);
    lr1_task2_line(draw_line_interpolation, lr1_result_folder / "t2_img_line_interp_autocount", &codec);
    lr1_task2_line(draw_line_interpolation_xloop, lr1_result_folder / "t2_img_line_interp_xloop", &codec);
    lr1_task2_line(draw_line_interpolation_xloop_fixX, lr1_result_folder / "t2_img_line_interp_xloop_fixX", &codec);
    lr1_task2_line(draw_line_interpolation_xloop_fixXfixY, lr1_result_folder / "t2_img_line_interp_xloop_fixXfixY", &codec);
    lr1_task2_line(draw_line_dy, lr1_result_folder / "t2_img_line_dy", &codec);
    lr1_task2_line(draw_line_dy_rev1, lr1_result_folder / "t2_img_line_dy_rev1", &codec);
    lr1_task2_line(draw_line, lr1_result_folder / "t2_img_line_bresenham", &codec);

    lr1_task3_vertices(lr1_result_folder / "t3_vertices.txt", input_folder / "model.obj");
    lr1_task4_draw_vertices(1000, 1000, input_folder / "model.obj", lr1_result_folder / "t4_draw_vertices", &codec);
    lr1_task5_polygons(lr1_result_folder / "t5_polygons.txt", input_folder / "model.obj");
    lr1_task6_draw_object(1000, 1000, input_folder / "model.obj", lr1_result_folder / "t6_object", &codec);

    #endif

    #ifdef LR2
    lr2_task9_single_triag(lr2_result_folder / "t9_single_triag", &codec);
    lr2_task9_single_triag_outofbound(lr2_result_folder / "t9_single_triag_outofbound", &codec);
    lr2_task9_single_triag_fulloutofbound(lr2_result_folder / "t9_single_triag_fulloutofbound", &codec);
    lr2_task9_multiple_triags_big(lr2_result_folder / "t9_multiple_triags_big", &codec);
    #endif

    codec.~image_codec();
    std::cout << "that's it" << std::endl;
}

template<typename E>
void draw_border(matrix_color<E>& img_matrix, E border_color)
{
    unsigned int vert_boundary = (int)img_matrix.height/10;
    unsigned int horiz_boundary = (int)img_matrix.width/10;

    for (size_t i = 0; i < img_matrix.height; i++)
    {
        for (size_t j = 0; j < img_matrix.width; j++)
        {
            if (i < vert_boundary || i > img_matrix.height - vert_boundary)
            {
                img_matrix.set(j, i, border_color);
            }
            else if (j < horiz_boundary || j > img_matrix.width - horiz_boundary)
            {
                img_matrix.set(j, i, border_color);   
            }
        }
    }
}

void decode_encode_img(std::string filepath, image_codec* codec)
{
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);

    ImageInfo info = codec->read_info(&img_buffer);
    
    matrix* mat;
    if (info.colorScheme == ImageColorScheme::IMAGE_RGB)
    {
        matrix_rgb* color_mat = new matrix_rgb(info.width, info.height);
        mat = color_mat;

        codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);
        draw_border<color_rgb>(*color_mat, color_rgb(255, 255, 255));
    }
    else
    {
        matrix_gray* color_mat = new matrix_gray(info.width, info.height);
        mat = color_mat;

        codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);

        draw_border<unsigned char>(*color_mat, 255);
    }

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / filepath);

    //delete mat;
}


void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    unsigned count = 1000;
    draw_line_interpolation_count(matrix, from, to, count, line_color);
}
