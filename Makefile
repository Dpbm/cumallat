TARGET=./build/mallat
TARGET_TESTS=./build/mallat_tests
SOURCE=mallat.cu
TEST=tests.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	cmake -B build
	cmake --build build
	$(TARGET)

debug: $(SOURCE) $(TEST)
	cmake -B build -DCMAKE_BUILD_TYPE=Debug
	cmake --build build 
	$(TARGET_TESTS)

