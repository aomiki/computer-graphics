#include <fstream>
#include <string> 
#include <regex>
#include <vector>
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include "obj_parser.h"

using namespace std;

bool parseVertex(string source, vertices* verts, unsigned i)
{
    const regex r_ver ("^v\\s+(-?[0-9]+(?:\\.[0-9]+)?) (-?[0-9]+(?:\\.[0-9]+)?) (-?[0-9]+(?:\\.[0-9]+)?)\\s*$");
    smatch m;
    if (regex_search(source, m, r_ver))
    {
        verts->x[i] = stof(m[1]);
        verts->y[i] = stof(m[2]);
        verts->z[i] = stof(m[3]);

        return true;
    }
    return false;
}   

bool parsePolygon(string source, polygons* polys, unsigned i) 
{
    const regex pol (R"(^f\s+([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)? ([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)? ([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)?(?: ([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)?)?\s*$)");
    smatch m;
    if (regex_search(source, m, pol))
    {
        polys->vertex_index1[i] = stoi(m[1]);
        polys->vertex_index2[i] = stoi(m[4]);
        polys->vertex_index3[i] = stoi(m[7]);

        return true;
    }
    return false;
}

enum ObjElement
{
    None = 0,
    Vertex,
    Polygon,
    Texture,
    Normal,
    ParameterSpace,
    LineEl,
    Comment
};

inline ObjElement getType(const char* str)
{
    ObjElement el = None;
    for (int i = 0; str[i]; i++)
    {
        if (std::isspace(str[i]))
        {
            if (el == None)
                continue;
            
            break;
        }

        switch (str[i])
        {
            case 'v':
                if (el == None)
                {
                    el = Vertex;
                    continue;
                }
                else
                {
                    return None;
                }
            case 'f':
                if (el == None)
                {
                    el = Polygon;
                    continue;
                }
                else
                {
                    return None;
                }
            case 'n':
                if (el == Vertex)
                {
                    el = Normal;
                    continue;
                }
                else
                {
                    return None;
                }
            case 't':
                if (el == Vertex)
                {
                    el = Texture;
                    continue;
                }
                else
                {
                    return None;
                }
            case 'p':
                if (el == Vertex)
                {
                    el = ParameterSpace;
                    continue;;
                }
                else
                {
                    return None;
                }
            case '#':
                return Comment;
            default:
                return None;
        }
    }

    return el;
}

struct ObjLine
{
    unsigned int i;
    ObjElement elementType;
    std::string line;
};

void processVertexLine(tbb::concurrent_queue<ObjLine>* queue, std::atomic<bool>* stop_token, vertices* verts = nullptr, polygons* polys = nullptr)
{
    ObjLine lineInfo;

    while (true)
    {
        if (queue->try_pop(lineInfo))
        {
            switch (lineInfo.elementType)
            {
                case ObjElement::Vertex:
                {
                    if (verts != nullptr)
                    {
                        parseVertex(lineInfo.line, verts, lineInfo.i);
                    }
                    break;   
                }
                case ObjElement::Polygon:
                {
                    if (polys != nullptr)
                    {
                        parsePolygon(lineInfo.line, polys, lineInfo.i);
                    }
                    break;
                }
            default:
                break;
            }
        }
        else if (*stop_token)
        {
            break;
        }
    }
}

void readObj(const string filename, vertices* verts, polygons* polys)
{
    if ((verts == nullptr) && (polys == nullptr))
        return;

    ifstream in(filename);
    string line;

    unsigned long n_vert = 0;
    unsigned long n_poly = 0;
    while (getline(in, line))
    {
        ObjElement el = getType(line.data());

        switch (el)
        {
            case ObjElement::Vertex:
                n_vert++;
                break;
            case ObjElement::Polygon:
                n_poly++;
                break;
            default:
                break;
        }
    }

    in.clear();
    in.seekg(0);

    verts->size = n_vert;
    polys->size = n_poly;

    float* verts_arr = new float[n_vert * 3];
    verts->x = verts_arr;
    verts->y = verts_arr + n_vert;
    verts->z = verts_arr + n_vert * 2;

    unsigned* polys_arr = new unsigned[n_poly * 3];
    polys->vertex_index1 = polys_arr;
    polys->vertex_index2 = polys_arr + n_poly;
    polys->vertex_index3 = polys_arr + n_poly*2;
    
    tbb::concurrent_queue<ObjLine> queue;
    std::atomic<bool> stop_token = false;

    int tMax = thread::hardware_concurrency();
    std::vector<std::thread> threads(tMax);
    for (int i = 0; i < tMax; i++)
        threads[i] = std::thread(processVertexLine, &queue, &stop_token, verts, polys);

    unsigned long vert_i = 0, poly_i = 0;
    while (getline(in, line))
    {
        ObjElement el = getType(line.data());

        ObjLine lineinfo;

        switch (el)
        {
            case Vertex:
                lineinfo.i = vert_i;
                vert_i++;
                break;
            case Polygon:
                lineinfo.i = poly_i;
                poly_i++;
                break;
            default:
                break;
        }

        lineinfo.elementType = el;
        lineinfo.line = line;
        queue.push(lineinfo);
    }

    stop_token = true;

    for (int i = 0; i < tMax; i++)
        threads[i].join();

    in.close();
}
