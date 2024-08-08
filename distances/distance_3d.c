#include <stdio.h>
#include <math.h>

typedef struct {
    float x;
    float y;
    float z;
} Point3D;

float euclidean_distance_3d(Point3D p1, Point3D p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

int main() {
    Point3D point1 = {1.0, 2.0, 3.0};
    Point3D point2 = {4.0, 6.0, 8.0};

    float distance = euclidean_distance_3d(point1, point2);

    printf("Euclidean distance from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f): %.2f\n",
           point1.x, point1.y, point1.z, point2.x, point2.y, point2.z, distance);

    return 0;
}