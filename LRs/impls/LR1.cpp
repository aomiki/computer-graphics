#include "LR1.h"
#include "image_draw_lines.h"
#include "image_draw_objects.h"
#include "obj_parser.h"
#include <fstream>
#include <vector>
#include <string>
#include <cmath> 

using namespace std;

void lr1_task2_line(void (&draw_line)(matrix_rgb* matrix, matrix_coord from, matrix_coord to, color_rgb line_color), std::string filepath, image_codec* codec)
{
    std::vector<unsigned char> png;
    matrix_rgb matrix(200, 200);
    color_rgb line_color(249, 89, 255);
    matrix_coord from(100, 100);

    matrix.fill(color_rgb(95, 164, 237));

    draw_star(&matrix, from, line_color, draw_line);

    codec->encode(&png, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    codec->save_image_file(&png, filepath);
}

void img_fill_gradient(matrix_rgb* matrix, color_rgb basecolor)
{
    for (size_t i = 0; i < matrix->width; i++)
    {
        for (size_t j = 0; j < matrix->height; j++)
        {
            basecolor.green = j;
            matrix->set(i, j, basecolor);
        }
    }
}

void lr1_task1_img_black(unsigned width, unsigned height, std::string filepath, image_codec* codec)
{
    matrix_gray matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(0);

    codec->encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
    codec->save_image_file(&png_buffer, filepath);
}

void lr1_task1_img_white(unsigned width, unsigned height, std::string filepath, image_codec* codec)
{
    matrix_gray matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(255);

    codec->encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
    codec->save_image_file(&png_buffer, filepath);
}

void lr1_task1_img_red(unsigned width, unsigned height, std::string filepath, image_codec* codec)
{
    matrix_rgb matrix(width, height);
    std::vector<unsigned char> png_buffer;

    matrix.fill(color_rgb(255, 0, 0));

    codec->encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    codec->save_image_file(&png_buffer, filepath);
}

void lr1_task1_img_gradient(unsigned width, unsigned height, std::string filepath, image_codec* codec)
{
    matrix_rgb matrix(width, height);
    std::vector<unsigned char> png_buffer;

    img_fill_gradient(&matrix, color_rgb(95, 0, 237));
    codec->encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    codec->save_image_file(&png_buffer, filepath);
}

void lr1_task3_vertices(std::string filepath, std::string in_filename)
{
    vertices verts;
    readObj(in_filename, &verts);
    ofstream out(filepath);
    for (size_t i = 0; i < verts.size; i++)
    {
        out << verts.x[i] << " " << verts.y[i] << " " << verts.z[i] << endl;
    }
    out.close();

    delete [] verts.x;
}

void lr1_task5_polygons(std::string filepath, std::string in_filename)
{
    polygons polys;
    readObj(in_filename, nullptr, &polys);
    ofstream out(filepath);
    for (size_t i = 0; i < polys.size; i++)
    {
        out << polys.vertex_index1[i] << " " << polys.vertex_index2[i] << " " << polys.vertex_index3[i] << endl;
    }
    out.close();

    delete [] polys.vertex_index1;
}

void lr1_task4_draw_vertices(unsigned width, unsigned height, std::string in_filename,std::string filepath, int scale, int offset, image_codec* codec, vertex_transforms* vt_transforms)
{
    matrix_rgb matrix(width, height);
    matrix.fill(color_rgb(255, 255, 255));
    vertices verts;
    polygons polys;
    readObj(in_filename, &verts);

    model_renderer renderer(&verts, &polys, vt_transforms);

    renderer.draw_vertices(&matrix, color_rgb(95, 0, 237), scale, offset);

    std::vector<unsigned char> png_buffer;
    codec->encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    codec->save_image_file(&png_buffer, filepath);

    delete [] verts.x;
}

void lr1_task6_draw_object(unsigned width, unsigned height, std::string in_filename,std::string filepath, image_codec* codec)
{
    matrix_rgb matrix(width, height);
    matrix.fill(color_rgb(255, 255, 255));
    vertices verts;
    polygons polys;
    readObj(in_filename, &verts, &polys);
    for (size_t i = 0; i < polys.size; i++)
    {
        matrix_coord v1{
            static_cast<unsigned>(std::round(5000*verts.x[polys.vertex_index1[i]-1] + 500)),
            static_cast<unsigned>(std::round(height - (5000*verts.y[polys.vertex_index1[i]-1] + 500)))
        };
        matrix_coord v2{
            static_cast<unsigned>(std::round(5000*verts.x[polys.vertex_index2[i]-1] + 500)),
            static_cast<unsigned>(std::round(height - (5000*verts.y[polys.vertex_index2[i]-1] + 500)))
        };
        matrix_coord v3{
            static_cast<unsigned>(std::round(5000*verts.x[polys.vertex_index3[i]-1] + 500)),
            static_cast<unsigned>(std::round(height - (5000*verts.y[polys.vertex_index3[i]-1] + 500)))
        };

        draw_line(&matrix, v1, v2, color_rgb(95, 0, 237));
        draw_line(&matrix, v2, v3, color_rgb(95, 0, 237));
        draw_line(&matrix, v3, v1, color_rgb(95, 0, 237));
    }

    std::vector<unsigned char> png_buffer;
    codec->encode(&png_buffer, &matrix, ImageColorScheme::IMAGE_RGB, 8);
    codec->save_image_file(&png_buffer, filepath);

    delete [] verts.x;

    delete [] polys.vertex_index1;
}
