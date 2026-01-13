#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

SDL_Window *window = nullptr;
SDL_GLContext glContext = nullptr;
GLuint program = 0;
GLuint VAO = 0, VBO = 0, colorVBO = 0;
int vertexCount = 36;
float rotationAngle = 0.0f;

GLuint compileShader(const std::string &source, GLenum type)
{
    GLuint shader = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n"
                  << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool initializeOpenGL()
{
    const std::string vertexShaderSrc =
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec3 aPosition;\n"
        "in vec3 aColor;\n"
        "uniform mat4 uModel;\n"
        "uniform mat4 uView;\n"
        "uniform mat4 uProjection;\n"
        "out vec3 vertexColor;\n"
        "void main() {\n"
        "    gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);\n"
        "    vertexColor = aColor;\n"
        "}";

    const std::string fragmentShaderSrc =
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec3 vertexColor;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(vertexColor, 1.0);\n"
        "}";

    GLuint vertexShader = compileShader(vertexShaderSrc, GL_VERTEX_SHADER);
    if (!vertexShader)
        return false;

    GLuint fragmentShader = compileShader(fragmentShaderSrc, GL_FRAGMENT_SHADER);
    if (!fragmentShader)
    {
        glDeleteShader(vertexShader);
        return false;
    }

    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program linking failed:\n"
                  << infoLog << std::endl;
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    std::cout << "✓ Shaders compiled" << std::endl;

    // 立方体顶点数据 - 每个面 6 个顶点（2 个三角形，逆时针绕序）
    float vertices[] = {
        // 前面 (z = 0.5)
        -0.5f, -0.5f, 0.5f,
        0.5f, -0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        -0.5f, -0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        -0.5f, 0.5f, 0.5f,

        // 后面 (z = -0.5)
        0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, 0.5f, -0.5f,
        0.5f, -0.5f, -0.5f,
        -0.5f, 0.5f, -0.5f,
        0.5f, 0.5f, -0.5f,

        // 上面 (y = 0.5)
        -0.5f, 0.5f, -0.5f,
        -0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        -0.5f, 0.5f, -0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, -0.5f,

        // 下面 (y = -0.5)
        -0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, 0.5f,
        -0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, 0.5f,
        -0.5f, -0.5f, 0.5f,

        // 右面 (x = 0.5)
        0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, -0.5f, -0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, -0.5f,

        // 左面 (x = -0.5)
        -0.5f, -0.5f, 0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, 0.5f, -0.5f,
        -0.5f, -0.5f, 0.5f,
        -0.5f, 0.5f, -0.5f,
        -0.5f, 0.5f, 0.5f};

    float colors[] = {
        // 前面 - 红色
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,

        // 后面 - 绿色
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,

        // 上面 - 蓝色
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,

        // 下面 - 黄色
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,

        // 右面 - 青色
        0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f,

        // 左面 - 洋红
        1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f};

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &colorVBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    GLint posAttrib = glGetAttribLocation(program, "aPosition");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 12, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
    GLint colorAttrib = glGetAttribLocation(program, "aColor");
    glEnableVertexAttribArray(colorAttrib);
    glVertexAttribPointer(colorAttrib, 3, GL_FLOAT, GL_FALSE, 12, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, 800, 600);

    return true;
}

bool handleEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT ||
            (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            return false;
    }
    return true;
}

void render(float deltaTime)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program);

    rotationAngle += 45.0f * deltaTime;

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(rotationAngle),
                                  glm::vec3(1.0f, 1.0f, 1.0f));
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f),
                                 glm::vec3(0.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f,
                                            0.1f, 100.0f);

    glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 1, GL_FALSE,
                       glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(program, "uView"), 1, GL_FALSE,
                       glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(program, "uProjection"), 1, GL_FALSE,
                       glm::value_ptr(projection));

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}

void cleanup()
{
    if (VAO)
        glDeleteVertexArrays(1, &VAO);
    if (VBO)
        glDeleteBuffers(1, &VBO);
    if (colorVBO)
        glDeleteBuffers(1, &colorVBO);
    if (program)
        glDeleteProgram(program);
    if (glContext)
        SDL_GL_DeleteContext(glContext);
    if (window)
        SDL_DestroyWindow(window);
    SDL_Quit();
}

int main(int argc, char *argv[])
{
    std::cout << "\nANGLE - OpenGL ES Demo\n"
              << std::endl;

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        return std::cerr << "SDL init failed\n", 1;

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    window = SDL_CreateWindow("ANGLE - OpenGL ES Demo", SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED, 800, 600,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!window)
        return std::cerr << "Window creation failed\n", cleanup(), 1;

    glContext = SDL_GL_CreateContext(window);
    if (!glContext)
        return std::cerr << "GL context creation failed\n", cleanup(), 1;

    SDL_GL_SetSwapInterval(1);

    std::cout << "OpenGL ES " << glGetString(GL_VERSION) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n"
              << std::endl;

    if (!initializeOpenGL())
        return std::cerr << "OpenGL init failed\n", cleanup(), 1;

    std::cout << "Press ESC to exit\n"
              << std::endl;

    Uint32 lastTime = SDL_GetTicks();
    while (handleEvents())
    {
        Uint32 currentTime = SDL_GetTicks();
        float deltaTime = (currentTime - lastTime) / 1000.0f;
        lastTime = currentTime;

        render(deltaTime);
        SDL_GL_SwapWindow(window);
    }

    cleanup();
    std::cout << "\nDone\n"
              << std::endl;
    return 0;
}
