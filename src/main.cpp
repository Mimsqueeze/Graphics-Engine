#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <iostream>

using namespace std;

// Define screen size
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

// Define number of triangles
const int NUM_TRIANGLES = 1;

// Define the vertex data for two triangles
GLfloat vertices[NUM_TRIANGLES * 3 * 3] = {
    // Triangle 1
    -0.5f, -0.5f, 0.0f,   // Bottom-left
     0.5f, -0.5f, 0.0f,   // Bottom-right
     0.0f,  0.5f, 0.0f,   // Top-center (higher z-coordinate)
};

#undef main
int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return -1;
    }

    // Set OpenGL version (2.1) and core profile
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    // Create a window
    SDL_Window* window = SDL_CreateWindow("SDL2 and GLEW", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        cerr << "SDL_CreateWindow Error: " << SDL_GetError() << endl;
        SDL_Quit();
        return -1;
    }

    // Create an OpenGL context
    SDL_GLContext context = SDL_GL_CreateContext(window);
    if (context == nullptr) {
        cerr << "SDL_GL_CreateContext Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW" << endl;
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Main loop
    bool running = true;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Enable client states
        glEnableClientState(GL_VERTEX_ARRAY);

        // Provide vertex data
        glVertexPointer(3, GL_FLOAT, 0, vertices);

        // Draw the two triangles
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
