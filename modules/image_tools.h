#include <vector>

#ifndef image_tools_h
#define image_tools_h

/// @brief Element of RGB matrix
struct color_rgb {
    color_rgb(unsigned char r, unsigned char g, unsigned char b)
    {
        red = r;
        green = g;
        blue = b;
    }

    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

struct matrix_coord {

    matrix_coord(unsigned x, unsigned y)
    {
        this->x = x;
        this->y = y;
    }

    unsigned x;
    unsigned y;
};

/// @brief Abstract matrix
class matrix {
    private:
        unsigned int get_interlaced_index(unsigned int x, unsigned int y);
    public:
        const unsigned int COMPONENTS_NUM;
        std::vector<unsigned char> array;
        unsigned width;
        unsigned height;

        matrix(unsigned int components_num, unsigned width, unsigned height);
        matrix(unsigned int components_num);

        unsigned char* get(unsigned int x, unsigned int y);
        unsigned char* get_c_arr_interlaced();
        unsigned int size_interlaced();
};

/// @brief Abstract image matrix
/// @tparam E Type of matrix elements
template<typename E>
class matrix_color : public matrix {
    private:
        void virtual element_to_c_arr(unsigned char* buffer, E value) = 0;
        E virtual c_arr_to_element(unsigned char* buffer) = 0;
    public:
        matrix_color(unsigned int components_num) : matrix(components_num) {}
        matrix_color(unsigned int components_num, unsigned width, unsigned height) : matrix(components_num, width, height) {}

        /// @brief Assign value to matrix cell
        /// @param[in] x x coordinate
        /// @param[in] y y coordinate
        /// @param[in] color element value
        void virtual set(unsigned x, unsigned y, E color);

        /// @brief Assign value to matrix cell
        /// @param[in] coord coordinates
        /// @param[in] color element value
        void virtual set(matrix_coord coord, E color);

        /// @brief Get matrix cell value
        /// @param[in] x x coordinate
        /// @param[in] y y coordinate
        /// @return cell value
        E virtual get(unsigned x, unsigned y);

        /// @brief Assign \p value to each matrix cell
        /// @param[in] value 
        void virtual fill(E value);
};

/// @brief Matrix for storing RGB images
class matrix_rgb : public matrix_color<color_rgb>
{
    private:
        void virtual element_to_c_arr(unsigned char* buffer, color_rgb value);
        color_rgb virtual c_arr_to_element(unsigned char* buffer);
    public:
        matrix_rgb(): matrix_color<color_rgb>(3) {}
        matrix_rgb(unsigned width, unsigned height): matrix_color<color_rgb>(3, width, height) {}
};

/// @brief Matrix for storing grayscale images
class matrix_gray : public matrix_color<unsigned char>
{
    private:
        void virtual element_to_c_arr(unsigned char* buffer, unsigned char value);
        unsigned char virtual c_arr_to_element(unsigned char* buffer);
    public:
        matrix_gray(unsigned width, unsigned height): matrix_color<unsigned char>(1, width, height) {}
};

#include "impls/image_tools.ipp"

#endif
