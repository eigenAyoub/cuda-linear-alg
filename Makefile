NVCC = nvcc
NVCC_FLAGS = -std=c++11
NVCC_FLAGS += -O3

INCLUDES = -I/usr/include -I/usr/include/x86_64-linux-gnu

LPATHS = -L/usr/lib/x86_64-linux-gnu

LIBS = -lcudart -lcudnn

SOURCES = forward.cu backprop.cu

OBJECTS = $(SOURCES:.cu=.o)

EXECUTABLE = softmax_example

all: $(EXECUTABLE)

# Rule to compile .cu files to .o files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Rule to compile .cpp files to .o files (if you have any)
# %.o: %.cpp
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Rule to link object files into the executable
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(LPATHS) $(OBJECTS) $(LIBS) -o $@

# Clean target
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Phony targets (targets that are not actual files)
.PHONY: all clean