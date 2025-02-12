#include "image_tools.h"
#include <string>

void lr1_task1_img_black(unsigned width, unsigned height, std::string filepath);
void lr1_task1_img_white(unsigned width, unsigned height, std::string filepath);
void lr1_task1_img_red(unsigned width, unsigned height, std::string filepath);
void lr1_task1_img_gradient(unsigned width, unsigned height, std::string filepath);

void lr1_task2_line(void (&draw_line)(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color), std::string filepath);
