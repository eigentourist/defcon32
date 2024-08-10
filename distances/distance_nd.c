#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float *coordinates;
    int dimensions;
} PointND;

float euclidean_distance_nd(PointND p1, PointND p2) {
    if (p1.dimensions != p2.dimensions) {
        printf("Error: Points have different dimensions.\n");
        exit(1);
    }
    float sum = 0.0;
    for (int i = 0; i < p1.dimensions; i++) {
        float diff = p2.coordinates[i] - p1.coordinates[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void print_point(PointND point) {
    printf("Point coordinates: ");
    for (int i = 0; i < point.dimensions; i++) {
        printf("%.2f", point.coordinates[i]);
        if (i < point.dimensions - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

void create_cursed_points(int dimensions) {
    if (dimensions < 2) {
        printf("Error: Dimensions should be at least 2.\n");
        return;
    }

    PointND point1, point2;
    point1.dimensions = point2.dimensions = dimensions;
    point1.coordinates = (float*) malloc(dimensions * sizeof(float));
    point2.coordinates = (float*) malloc(dimensions * sizeof(float));

    // Initialize points: Close on 30 axes, distant on 2
    for (int i = 0; i < dimensions - 2; i++) {
        point1.coordinates[i] = (float)i + 0.1f;
        point2.coordinates[i] = (float)i + 0.2f;
    }
    // Set two axes with a large distance
    point1.coordinates[dimensions - 2] = 50.0;
    point2.coordinates[dimensions - 2] = 500.0;
    point1.coordinates[dimensions - 1] = 60.0;
    point2.coordinates[dimensions - 1] = 600.0;

    printf("\nCursed 32D Points:\n");
    print_point(point1);
    print_point(point2);

    float distance = euclidean_distance_nd(point1, point2);
    printf("Euclidean distance between cursed 32D points: %.2f\n", distance);

    // Free allocated memory
    free(point1.coordinates);
    free(point2.coordinates);
}

int main() {
    PointND point1, point2;
    int dimensions = 3;

    // Allocate memory for 3D coordinates
    point1.dimensions = point2.dimensions = dimensions;
    point1.coordinates = (float*) malloc(dimensions * sizeof(float));
    point2.coordinates = (float*) malloc(dimensions * sizeof(float));

    // Initialize points close on 2 axes, far on 1
    point1.coordinates[0] = 1.0;
    point1.coordinates[1] = 2.0;
    point1.coordinates[2] = 3.0;

    point2.coordinates[0] = 1.1;
    point2.coordinates[1] = 2.1;
    point2.coordinates[2] = 100.0;

    printf("3D Points:\n");
    print_point(point1);
    print_point(point2);

    float distance = euclidean_distance_nd(point1, point2);
    printf("Euclidean distance: %.2f\n", distance);

    // Free allocated memory
    free(point1.coordinates);
    free(point2.coordinates);

    // Uncomment the following line to calculate the distance for 32D cursed points
    // create_cursed_points(32);

    return 0;
}
