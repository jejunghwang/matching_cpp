# CMakeLists.txt

# 최소 CMake 버전 설정
cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름 및 사용 언어 설정
project(block_matching LANGUAGES CXX)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# OpenCV 패키지 찾기
set(Opencv_DIR /usr/local/Cellar/opencv/4.5.1_2/lib/cmake/opencv4)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# 포함 디렉토리 설정
include_directories(${OpenCV_INCLUDE_DIRS})

# 실행 파일 생성 (여기서는 main.cpp 사용)
add_executable(block_matching src/matching.cpp)

# OpenCV 라이브러리 링크
target_link_libraries(block_matching ${OpenCV_LIBS})


