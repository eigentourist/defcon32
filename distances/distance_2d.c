#include <stdio.h>
#include <math.h>

typedef struct {
    float x;
    float y;
} Point2D;

float euclidean_distance_2d(Point2D p1, Point2D p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return sqrt(dx*dx + dy*dy);
}

int main() {
    Point2D point1 = {1.0, 2.0};
    Point2D point2 = {4.0, 6.0};

    float distance = euclidean_distance_2d(point1, point2);

    printf("Euclidean distance from (%.2f, %.2f) to (%.2f, %.2f): %.2f\n",
           point1.x, point1.y, point2.x, point2.y, distance);

    return 0;
}