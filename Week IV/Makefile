# Compiler yang digunakan
CXX := clang++

# Flags untuk kompilasi
CXXFLAGS := -O2 -mno-avx512f -march=native
LFLAGS := -fopenmp
# Nama target
TARGET := output.exe

# Sumber file
SOURCES := main.cc

# Aturan untuk membuat target executable
$(TARGET): $(SOURCES)
	$(CXX) -o $(TARGET) $(SOURCES) $(CXXFLAGS) $(LFLAGS)

# Aturan untuk membersihkan file objek dan executable
clean:
	rm -f $(TARGET)

# Menggunakan phony targets untuk target yang tidak menghasilkan file
.PHONY: clean