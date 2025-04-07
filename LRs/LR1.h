#include "vertex_tools.h"
#include "image_codec.h"
#include <string>

void lr1_task1_img_black(unsigned width, unsigned height, std::string filepath, image_codec* codec);
void lr1_task1_img_white(unsigned width, unsigned height, std::string filepath, image_codec* codec);
void lr1_task1_img_red(unsigned width, unsigned height, std::string filepath, image_codec* codec);
void lr1_task1_img_gradient(unsigned width, unsigned height, std::string filepath, image_codec* codec);

void lr1_task2_line(void (&draw_line)(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color), std::string filepath, image_codec* codec);
void lr1_task3_vertices(std::string filepath, std::string in_filename);
void lr1_task5_polygons(std::string filepath, std::string in_filename);
void lr1_task4_draw_vertices(unsigned width, unsigned height, std::string in_filename,std::string filepath, int scale, int offset, image_codec* codec, vertex_transforms* vt_transforms);
void lr1_task6_draw_object(unsigned width, unsigned height, std::string in_filename,std::string filepath, image_codec* codec);
