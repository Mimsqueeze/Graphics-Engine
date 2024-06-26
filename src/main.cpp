#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "test.hpp"

using namespace std;

// Define screen size
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

void print_matrix(const string& name, const float *matrix, int rows, int cols) {
    cout << name << ": " << endl;
    cout << fixed << setprecision(2);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[j * rows + i] << "\t";
        }
        cout << endl;
    }
}

#undef main
int main(int argc, char* argv[]) {
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);

    // Create a window
    SDL_Window* window = SDL_CreateWindow("SDL2 and Manual Projection", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

    // Create an OpenGL context
    SDL_GLContext context = SDL_GL_CreateContext(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    // Define number of triangles
    const int NUM_TRIANGLES = 1;

    // Define the vertex data for one triangle
    float vertices[NUM_TRIANGLES * 3 * 3] = {
        // Triangle 1
        -0.5f, -0.5f, 0.0f,   // Bottom-left (closer)
        0.5f, -0.5f, 0.0f,   // Bottom-right (closer)
        0.0f,  0.5f, 0.0f    // Top-center (farther)
    };

    print_matrix("Original vertices", vertices, 3, NUM_TRIANGLES * 3);

    // Define perspective projection parameters
    const float fov = 90.0f;    // Field of view
    const float aspect = 1280.0f / 720.0f; // Aspect ratio
    const float nearClip = 0.1f; // Near clipping plane
    const float farClip = 100.0f; // Far clipping plane

    float results[NUM_TRIANGLES * 3 * 3]{0};
    perspective_projection(results, vertices, NUM_TRIANGLES * 3, fov, aspect, nearClip, farClip);

    // Main loop
    bool running = true;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // Set viewport
        glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

        // Clear the color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Enable client states
        glEnableClientState(GL_VERTEX_ARRAY);

        // Provide vertex data
        glVertexPointer(3, GL_FLOAT, 0, vertices);

        // Draw the triangle
        glDrawArrays(GL_TRIANGLES, 0, 3 * NUM_TRIANGLES);

        // Disable client states
        glDisableClientState(GL_VERTEX_ARRAY);

        // Swap the buffers
        SDL_GL_SwapWindow(window);
    }

    // Clean up
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
