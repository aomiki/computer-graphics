#include <cmath> 
#include "LR2.h"
#include "image_draw_objects.h"

void lr2_task9_single_triag(std::string out_path, image_codec *codec)
{
    matrix_gray m(10, 10);

    m.fill(255);

    vertex v1(3, 1);
    vertex v2(5, 4);
    vertex v3(8, 1);
    
    unsigned char triag_color = 0;

    draw_polygon(&m, triag_color, v1, v2, v3);

    std::vector<unsigned char> img_buff;

    codec->encode(&img_buff, &m, ImageColorScheme::IMAGE_GRAY, 8);
    codec->save_image_file(&img_buff, out_path);
}

void lr2_task9_single_triag_outofbound(std::string out_path, image_codec *codec)
{
    matrix_gray m(10, 10);

    m.fill(255);

    vertex v1(6, 2);
    vertex v2(8, 5);
    vertex v3(13, 3);

    unsigned char triag_color = 0;

    draw_polygon(&m, triag_color, v1, v2, v3);

    std::vector<unsigned char> img_buff;

    codec->encode(&img_buff, &m, ImageColorScheme::IMAGE_GRAY, 8);
    codec->save_image_file(&img_buff, out_path);
}

void lr2_task9_single_triag_fulloutofbound(std::string out_path, image_codec *codec)
{
    matrix_gray m(10, 10);

    m.fill(255);

    vertex v1(11, 1);
    vertex v2(12, 5);
    vertex v3(13, 2);

    unsigned char triag_color = 0;

    draw_polygon(&m, triag_color, v1, v2, v3);

    std::vector<unsigned char> img_buff;

    codec->encode(&img_buff, &m, ImageColorScheme::IMAGE_GRAY, 8);
    codec->save_image_file(&img_buff, out_path);
}

void lr2_task9_multiple_triags_big(std::string out_path, image_codec *codec)
{
    matrix_gray m(1000, 1000);

    m.fill(255);

    vertex v1(300, 100);
    vertex v2(500, 400);
    vertex v3(800, 100);

    vertex v21(600, 200);
    vertex v22(800, 500);
    vertex v23(1300, 300);

    unsigned char triag_color = 0;

    draw_polygon(&m, triag_color, v1, v2, v3);
    draw_polygon(&m, triag_color, v21, v22, v23);

    std::vector<unsigned char> img_buff;

    codec->encode(&img_buff, &m, ImageColorScheme::IMAGE_GRAY, 8);
    codec->save_image_file(&img_buff, out_path);
}

void lr2_task10_model(std::string in_path, std::string out_path, std::vector<unsigned char>* png_buffer,  unsigned width, unsigned height, image_codec *codec, vertex_transforms* vt_transforms)
{
    matrix_gray matrix(width, height);
    matrix.fill(255);
    vertices verts;
    polygons polys;
    vertex v;
    readObj(in_path, &verts, &polys);
    
    float offsets[3] = {0, 0, 5};
    float angles[3] = {0, 3, 0}; 
    float scaleX = 200;
    float scaleY = 200;
    
    vt_transforms->rotateAndOffset(&verts, &verts, offsets, angles);

    unsigned char modelColor[3] = { 255, 255, 255 };

    draw_polygons_filled(&matrix, &verts, &polys, scaleX, scaleY, modelColor);
    codec->encode(png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);

    delete verts.x;
    delete verts.y;
    delete verts.z;

    delete polys.vertex_index1;
    delete polys.vertex_index2;
    delete polys.vertex_index3;
}
