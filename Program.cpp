#include "lodepng.h"
#include "image_codec.h"
#include "LR1.h"
#include "image_draw_lines.h"

#include <iostream>
#include <vector>
#include <string>

std::string result_folder = "output";
std::string lr1_result_folder = result_folder + "/LR1/";

void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);

int main()
{
    std::cout << "Shellow from SSAU!" << std::endl;

    lr1_task1_img_black(100, 100, lr1_result_folder + "t1_img_black.png");

    lr1_task1_img_white(100, 100, lr1_result_folder + "t1_img_white.png");

    lr1_task1_img_red(100, 100, lr1_result_folder + "t1_img_red.png");

    lr1_task1_img_gradient(255, 255, lr1_result_folder + "t1_img_gradient.png");
    
    lr1_task2_line(dotted_line_count, lr1_result_folder + "t2_img_line_interp_count.png");
    lr1_task2_line(draw_line_interpolation, lr1_result_folder + "t2_img_line_interp_autocount.png");
    lr1_task2_line(draw_line_interpolation_xloop, lr1_result_folder + "t2_img_line_interp_xloop.png");
    lr1_task2_line(draw_line_interpolation_xloop_fixX, lr1_result_folder + "t2_img_line_interp_xloop_fixX.png");
    lr1_task2_line(draw_line_interpolation_xloop_fixXfixY, lr1_result_folder + "t2_img_line_interp_xloop_fixXfixY.png");
    lr1_task2_line(draw_line_dy, lr1_result_folder + "t2_img_line_dy.png");
    lr1_task2_line(draw_line_dy_rev1, lr1_result_folder + "t2_img_line_dy_rev1.png");
    lr1_task2_line(draw_line, lr1_result_folder + "t2_img_line_bresenham.png");


    std::cout << "that's it" << std::endl;
}

void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color)
{
    unsigned count = 1000;
    draw_line_interpolation_count(matrix, from, to, count, line_color);
}

