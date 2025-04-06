#ifndef OBJ_PARSER_H
#define OBJ_PARSER_H

#include <string> 
#include <vector>

#include "image_tools.h"
using namespace std;

struct vertex {
     __shared_func__ vertex()
     {}

     __shared_func__ vertex(float x, float y, float z)
     {
          this->x = x;
          this->y = y;
          this->z = z;
     }

     __shared_func__ vertex(float x, float y)
     {
          this->x = x;
          this->y = y;
     }

     float x, y, z;
};

struct vertices{
     __shared_func__ vertices()
     {}

     float* x;
     float* y;
     float* z;

     unsigned size;
};

struct polygons{
     unsigned* vertex_index1;
     unsigned* vertex_index2;
     unsigned* vertex_index3;

     unsigned size;
};

/// @brief парсит только вершины из строки
/// @param source строка указательна на вершину
/// @return спарсило или нет?
bool parseVertex(string source, vertices* verts, unsigned i);

/// @brief парсит только полигоны из строки
/// @param source строка, указатель на на полигон
/// @return спарсило или нет ?
bool parsePolygon(string source, polygons* polys, unsigned i);

void readObj(const string filename, vertices* vertices = nullptr, polygons* polygons = nullptr);

#endif
