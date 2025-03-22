#include "image_codec.h"

#include <string>

void lr2_task9_single_triag(std::string out_path, image_codec* codec);

void lr2_task9_single_triag_outofbound(std::string out_path, image_codec* codec);

void lr2_task9_single_triag_fulloutofbound(std::string out_path, image_codec *codec);

void lr2_task9_multiple_triags_big(std::string out_path, image_codec *codec);

void lr2_task10_model(std::string in_path, std::string out_path, std::vector<unsigned char>* png_buffer, unsigned width, unsigned height, int scale, int offset, image_codec *codec);
