#ifndef OBJ_PARSER_H
#define OBJ_PARSER_H

#include <string> 
#include <vector>

#include "image_tools.h"
using namespace std;

struct vertex{
     __matrix_attr__ vertex()
     {}

     __matrix_attr__ vertex(double x, double y, double z = 0)
     {
          this->x = x;
          this->y = y;
          this->z = z;
     }

     double x;
     double y;
     double z;
};

struct polygon{
     unsigned vertex_index1;
     unsigned vertex_index2;
     unsigned vertex_index3;
};

/// @brief парсит только вершины из строки
/// @param source строка указательна на вершину
/// @return спарсило или нет?
bool parseVertex(string source, vertex* v);

/// @brief парсит только полигоны из строки
/// @param source строка, указатель на на полигон
/// @return спарсило или нет ?
bool parsePolygon(string source, polygon* p);

void readObj(const string filename, vector <vertex>* vertices = nullptr, vector <polygon>* polygons = nullptr);

#endif
