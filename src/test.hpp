#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void perspective_projection(float *result, float *vertices, int cols, float fov, float aspect, float nearClip, float farClip);