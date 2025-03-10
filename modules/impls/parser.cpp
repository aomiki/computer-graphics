#include <fstream>
#include <string> 
#include <regex>
#include <vector>
#include "obj_parser.h"
using namespace std;

bool parseVertex(string source, vertex* v)
{
    const regex r_ver ("^v( -?[0-9]+(?:\\.[0-9]+)?)( -?[0-9]+(?:\\.[0-9]+)?)( -?[0-9]+(?:\\.[0-9]+)?)\\s*$");
    smatch m;
    if (regex_search(source, m, r_ver))
    {
        v->x = stod(m[1]);
        v->y= stod(m[2]);
        v->z = stod(m[3]);

        return true;
    }
    return false;
}   

bool parsePolygon(string source, polygon* p) 
{
    const regex pol (R"(^f ([0-9]+)\/([0-9]+)\/([0-9]+) ([0-9]+)\/([0-9]+)\/([0-9]+) ([0-9]+)\/([0-9]+)\/([0-9]+)\s*$)");
    smatch m;
    if (regex_search(source, m, pol))
    {
        p->vertex_index1 = stoi(m[1]);
        p->vertex_index2 = stoi(m[4]);
        p->vertex_index3 = stoi(m[7]);

        return true;
    }
    return false;
}

void readObj(const string filename, vector <vertex>* vertices, vector <polygon>* polygons)
{
    if ((vertices == nullptr) && (polygons == nullptr))
    {
        return;
    }

    ifstream in(filename);
    string line;
    while (getline(in, line))
    {
        polygon p;
        if ((polygons != nullptr) && parsePolygon(line, &p))
        {
            polygons->push_back(p); 
        }

        vertex v;
        if ((vertices != nullptr) && parseVertex(line, &v))
        {
            vertices->push_back(v);
        }
    }

    in.close();
}
