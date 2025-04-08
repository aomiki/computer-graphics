#ifndef IMAGE_DRAW_OBJECTS_H
#define IMAGE_DRAW_OBJECTS_H
#include "vertex_tools.h"
#include "image_tools.h"
#include "obj_parser.h"
#include "image_codec.h"

template<typename E>
void draw_polygon(matrix_color<E>* img, E polyg_color, vertex v1, vertex v2, vertex v3);

template<typename E>
void draw_polygons(matrix_color<E>* img, vertices* verts, E polyg_color, vertex v1, vertex v2, vertex v3);


class model_renderer
{
    private:
        unsigned char* d_original_geometry_membuf;
        unsigned char* d_verts_transformed_membuf;

        polygons* d_polys;
        vertices* d_verts;
        unsigned n_verts;
        unsigned n_polys;

        vertices* d_verts_transformed;
        vertices h_verts_transformed_d_copy;

        image_codec* codec;
        
    public:
        model_renderer(vertices *verts, polygons *polys);
        ~model_renderer();

        template<typename E>
        void draw_polygons(matrix_color<E> *img, float scaleX, float scaleY, unsigned char* modelColor);

        template<typename E>
        void draw_vertices(matrix_color<E>* m, E vertex_color, float scaleX, float scaleY);

        void rotateAndOffset(float offsets[3], float angles[3], vertex_transforms* vt_transformer);

        unsigned get_vertices_size()
        {
            return n_verts;
        }
        
        unsigned get_polygons_size()
        {
            return n_polys;
        }
};

class scene
{
    private:
        image_codec* codec;
        vertex_transforms* vt_transformer;
        matrix* img_matrix;
        ImageColorScheme colorScheme;

    public:
        scene();
        ~scene();

        image_codec* get_codec()
        {
            return codec;
        }

        void set_scene_params(unsigned width, unsigned height, ImageColorScheme colorScheme);
        void fill(unsigned char* color);

        void encode(std::vector<unsigned char>& img_buffer);
        void draw_model_polygons(model_renderer& model, float scaleX, float scaleY, unsigned char* modelColor);
        void draw_model_vertices(model_renderer& model, float scaleX, float scaleY, unsigned char* modelColor);
        void transform_model(model_renderer& model, float offsets[3], float angles[3]);       
};

__shared_func__ void calc_triangle_boundaries(screen_coords& min_coord, screen_coords& max_coord, vec3& screen_v1, vec3& screen_v2, vec3& screen_v3, matrix& m);

#endif
