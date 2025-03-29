#include <fstream>
#include <string> 
#include <regex>
#include <vector>
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include "obj_parser.h"

using namespace std;

bool parseVertex(string source, vertex* v)
{
    const regex r_ver ("^v\\s+(-?[0-9]+(?:\\.[0-9]+)?) (-?[0-9]+(?:\\.[0-9]+)?) (-?[0-9]+(?:\\.[0-9]+)?)\\s*$");
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
    const regex pol (R"(^f\s+([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)? ([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)? ([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)?(?: ([0-9]+)(?:\/([0-9]+)?(?:\/([0-9]+))?)?)?\s*$)");
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

void processVertexLine(tbb::concurrent_queue<ObjLine>* queue, std::atomic<bool>* stop_token, std::vector<vertex>* vertices = nullptr, std::vector<polygon>* polygons = nullptr)
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
                    vertex v;
                    if ((vertices != nullptr) && parseVertex(lineInfo.line, &v))
                    {
                        (*vertices)[lineInfo.i] = v;
                    }
                    break;   
                }
                case ObjElement::Polygon:
                {
                    polygon p;
                    if ((polygons != nullptr) && parsePolygon(lineInfo.line, &p))
                    {
                        (*polygons)[lineInfo.i] = p; 
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

void readObj(const string filename, vector <vertex>* vertices, vector <polygon>* polygons)
{
    if ((vertices == nullptr) && (polygons == nullptr))
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

    vertices->resize(n_vert);
    polygons->resize(n_poly);
    
    tbb::concurrent_queue<ObjLine> queue;
    std::atomic<bool> stop_token = false;

    int tMax = thread::hardware_concurrency();
    std::vector<std::thread> threads(tMax);
    for (int i = 0; i < tMax; i++)
        threads[i] = std::thread(processVertexLine, &queue, &stop_token, vertices, polygons);

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
