#ifndef SM_HEADER
#define SM_HEADER
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <vector>

#include <string>
#include "constants.h"
using namespace std;

struct arguments
{
    int edge_count;
    long long count;
    double time;
    int degree;
    int vertices;
};

#ifdef useVertexAsStart
const int break_level = 0;
#else
const int break_level = 1;
#endif

struct arguments SubgraphMatching(int process_id, int process_num, struct arguments args, char *argv[]);
#endif