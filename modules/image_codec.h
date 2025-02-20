#include <string>
#include <vector>
#include "image_tools.h"

#ifndef image_codec_h
#define image_codec_h

enum ImageColorScheme{
    IMAGE_GRAY,
    IMAGE_RGB
};

class image_codec {
    public:
        image_codec();

        void encode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth);
        
        void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth);

        void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath);
        
        /// @brief Saves image to file
        /// @param png_buffer image data
        /// @param image_filepath filepath, without extension
        void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath);

        ~image_codec();
};

#endif
