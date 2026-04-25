#include"constants.h"

int intersection_size = 7;
int restriction_size = 1;
int *intersection_orders = new int[7]{0, 0, 1, -1, 2, -1, 3};
int *intersection_offset = new int[5]{0, 1, 3, 5, 7};
int *restriction = new int[1];
int *reuse = new int[5]{-1, -1, -1, 2, 3};
