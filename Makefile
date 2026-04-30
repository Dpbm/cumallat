TARGET=./build/mallat
SOURCE=mallat.cu

all: $(TARGET)


$(TARGET): $(SOURCE)
	cmake -B build
	cmake --build build
	$(TARGET)
