#include"constants.h"

int intersection_size = 5;
int restriction_size = 4;
int *intersection_orders = new int[5]{0, 0, 1, 1, 2};
int *intersection_offset = new int[4]{0, 1, 3, 5};
int *restriction = new int[4]{-1, -1, 1, 0};
int *reuse = new int[4];
