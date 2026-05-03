TARGET=./build/mallat
TARGET_TESTS=./build/mallat_tests
SOURCE=example.cu
TEST=tests.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
	cmake --build build
	$(TARGET)

debug: $(SOURCE) $(TEST)
	cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=native
	cmake --build build 
	$(TARGET_TESTS)

