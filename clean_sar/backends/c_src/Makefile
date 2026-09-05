NVCC ?= nvcc
NVCCFLAGS ?= -O3 -shared -Xcompiler -fPIC --std=c++14 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90

TARGET = libcleansar.so
SRCS = clean_sar_cuda.cu
HEADERS = clean_sar_cuda.h psf_math.cuh reduction.cuh

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
