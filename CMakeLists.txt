cmake_minimum_required(VERSION 3.16)
project(TemplateGLFW)

# 基本設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_compile_options(-Wno-volatile)

# ソースファイルの収集
file(GLOB SOURCES src/*.cpp src/*.h src/*.hpp src/*.glsl)

# 必須パッケージ
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(glfw3 REQUIRED)

# OpenCVパッケージ（必要な場合のみ）
find_package(OpenCV REQUIRED)
set(OpenCV_STATIC OFF)

# インクルードディレクトリの設定
include_directories(
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
)

# リンクディレクトリの設定
link_directories(${GLEW_LIBRARY_DIRS})

# 実行ファイルの作成
add_executable(${PROJECT_NAME} ${SOURCES})

# ライブラリのリンク
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    glfw
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# 実行ファイルの出力先設定
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
