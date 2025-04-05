#include "lodepng.h"
#include "image_codec.h"
#include "vertex_tools.h"
#include "LR1.h"
#include "LR2.h"
#include "image_draw_lines.h"
#include "image_tools.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "gui/mainwindow.h"

#include <QApplication>

#define LR2
#define LR_LATEST

namespace fs = std::filesystem;

const fs::path result_folder("output");
const fs::path input_folder ("input");
const fs::path lr1_result_folder = result_folder / "LR1";
const fs::path lr2_result_folder = result_folder / "LR2";


void dotted_line_count(matrix_rgb *matrix, matrix_coord from, matrix_coord to, color_rgb line_color);
void decode_encode_img(std::string filepath, image_codec* codec);

int main(int argc, char *argv[])
{
    std::cout << "Shellow from SSAU!" << std::endl;
    
    if (argc > 1 && strcmp(argv[1], "--gui") == 0)
    {
        QApplication a(argc, argv);
        MainWindow w;
        w.show();
        return a.exec();
    }

    image_codec codec;
    vertex_transforms vt_transforms;

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

    //lr1_task3_vertices(lr1_result_folder / "t3_vertices.txt", input_folder / "model.obj");
    lr1_task4_draw_vertices(1000, 1000, input_folder / "model.obj", lr1_result_folder / "t4_draw_vertices", 5000, 500, &codec);
    lr1_task5_polygons(lr1_result_folder / "t5_polygons.txt", input_folder / "model.obj");
    lr1_task6_draw_object(1000, 1000, input_folder / "model.obj", lr1_result_folder / "t6_object", &codec);

    #endif

    #ifdef VERTICES_TEST

    lr1_task4_draw_vertices(1000, 1000, input_folder / "tposegirl_pc.obj", lr1_result_folder / "tposegirl_pc", &codec);

    #endif

    #ifdef LR2
    #ifndef LR_LATEST
    lr2_task9_single_triag(lr2_result_folder / "t9_single_triag", &png_buffer, &codec, &vt_transforms);
    lr2_task9_single_triag_outofbound(lr2_result_folder / "t9_single_triag_outofbound", &png_buffer, &codec, &vt_transforms);
    lr2_task9_single_triag_fulloutofbound(lr2_result_folder / "t9_single_triag_fulloutofbound", &png_buffer, &codec, &vt_transforms);
    lr2_task9_multiple_triags_big(lr2_result_folder / "t9_multiple_triags_big", &png_buffer, &codec, &vt_transforms);
    lr1_task4_draw_vertices(500, 500, input_folder / "dagger.obj", lr2_result_folder / "dagger", &png_buffer, 100, 100, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "model.obj", lr2_result_folder / "t10_model_filled", &png_buffer, 1000, 1000, 5000, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "dagger.obj", lr2_result_folder / "t10_dagger_filled", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "clock.obj", lr2_result_folder / "t10_clock_filled", &png_buffer, 1000, 1000, 40, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "12268_banjofrog_v1_L3.obj", lr2_result_folder / "t10_banjofro_filled", &png_buffer, 1000, 1000, 40, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "modelv2.obj", lr2_result_folder / "t10_modelv2_filled", &png_buffer, 1000, 1000, 300, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "12221_Cat_v1_l3.obj", lr2_result_folder / "t10_cat_filled", &png_buffer, 1000, 1000, 10, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "genshin_teapot.obj", lr2_result_folder / "t10_genshin_teapot_filled", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "heart.obj", lr2_result_folder / "t10_heart_filled", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "animehead.obj", lr2_result_folder / "t10_animehead_filled", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "shield.obj", lr2_result_folder / "t10_shield_filled", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "boy_demon_red_eyes_horns_teeth_mask_red_skin_cartoon_style_draft.obj", &png_buffer, lr2_result_folder / "t10_boy_demon_red_eyes_horns_teeth_mask_red_skin_cartoon_style_draft_filled", 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "pickachu2.obj", lr2_result_folder / "pickachu2", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "aranara.obj", lr2_result_folder / "aranara", &png_buffer, 500, 500, 200, 270, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "Paimon.obj", lr2_result_folder / "Paimon", &png_buffer, 1000, 1000, 47, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "Rushia.obj", lr2_result_folder / "Rushia", &png_buffer, 3000, 3000, 1, 500, &codec, &vt_transforms);
    lr2_task10_model(input_folder / "nvLogo.obj", lr2_result_folder / "nvLogo", &png_buffer, 500, 500, 20, 270, &codec, &vt_transforms);
    #endif
    std::vector<unsigned char> png_buffer;
    lr2_task10_model(input_folder / "Paimon.obj", lr2_result_folder / "Paimon", &png_buffer, 1000, 1000, &codec, &vt_transforms);
    #endif

    codec.save_image_file(&png_buffer, lr2_result_folder / "Paimon");

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
