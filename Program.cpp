#include "lodepng.h"
#include "image_codec.h"
#include "LR1.h"
#include "image_draw_lines.h"


#include <iostream>
#include <vector>
#include <string>

const std::string result_folder = "output";
const std::string input_folder = "input";
const std::string lr1_result_folder = result_folder + "/LR1/";


void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void decode_encode_img(std::string filepath);

int main()
{
    std::cout << "Shellow from SSAU!" << std::endl;

    decode_encode_img(input_folder+"/senku.png");

    lr1_task1_img_black(4000, 2000, lr1_result_folder + "t1_img_black");

    lr1_task1_img_white(4000, 2000, lr1_result_folder + "t1_img_white");

    lr1_task1_img_red(4000, 2000, lr1_result_folder + "t1_img_red");

    lr1_task1_img_gradient(255, 255, lr1_result_folder + "t1_img_gradient");
    
    lr1_task2_line(dotted_line_count, lr1_result_folder + "t2_img_line_interp_count");
    lr1_task2_line(draw_line_interpolation, lr1_result_folder + "t2_img_line_interp_autocount");
    lr1_task2_line(draw_line_interpolation_xloop, lr1_result_folder + "t2_img_line_interp_xloop");
    lr1_task2_line(draw_line_interpolation_xloop_fixX, lr1_result_folder + "t2_img_line_interp_xloop_fixX");
    lr1_task2_line(draw_line_interpolation_xloop_fixXfixY, lr1_result_folder + "t2_img_line_interp_xloop_fixXfixY");
    lr1_task2_line(draw_line_dy, lr1_result_folder + "t2_img_line_dy");
    lr1_task2_line(draw_line_dy_rev1, lr1_result_folder + "t2_img_line_dy_rev1");
    lr1_task2_line(draw_line, lr1_result_folder + "t2_img_line_bresenham");

    std::cout << "that's it" << std::endl;

    lr1_task3_vertices(lr1_result_folder + "t3_vertices.txt", input_folder + "/" + "model.obj");
    lr1_task4_draw_vertices(1000, 1000, input_folder + "/" + "model.obj", lr1_result_folder + "t4_draw_vertices");
    lr1_task5_polygons(lr1_result_folder + "t5_polygons.txt", input_folder + "/" + "model.obj");
    lr1_task6_draw_object(1000, 1000, input_folder + "/" + "model.obj", lr1_result_folder + "t6_object");
    

}

void decode_encode_img(std::string filepath)
{
    std::vector<unsigned char> img_buffer;

    load_image_file(&img_buffer, filepath);

    matrix_rgb img_matrix;
    decode(&img_buffer, &img_matrix,ImageColorScheme::IMAGE_RGB, 8);

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
    encode(&img_buffer, &img_matrix, ImageColorScheme::IMAGE_RGB, 8);

    save_image_file(&img_buffer, result_folder+"/"+filepath);
}

void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    unsigned count = 1000;
    draw_line_interpolation_count(matrix, from, to, count, line_color);
}


