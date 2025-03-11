.PHONY: clean clean-output-LR1 clean-output-LR2

#Main compiler
CXX = g++

#Modules
MODULES_SRC = $(wildcard modules/impls/*.cpp)
MODULES := $(patsubst %.cpp,%.o,$(MODULES_SRC))

#Modules that are used both by LODE implementation and directly in CUDA kernels
MODULES_SHARED_SRC = $(wildcard modules/impls_shared/*.cpp)
MODULES_SHARED_CPP := $(patsubst %.cpp,%.o,$(MODULES_SHARED_SRC))
MODULES_SHARED_CUDA := $(patsubst %.cpp,%.cu.o,$(MODULES_SHARED_SRC))
MODULES_SHARED_CUDA_LINKED := $(patsubst %,%.linked.o,$(MODULES_SHARED_SRC))

#Laboratory works
LRS_SRC := $(wildcard LRs/impls/*.cpp)
LRS := $(patsubst %.cpp,%.o,$(LRS_SRC))

#LodePNG implementation
LODE_SRC := $(wildcard include/lodepng/lodepng.cpp modules/impls_cpu/*.cpp)
LODE :=  $(patsubst %.cpp,%.o,$(LODE_SRC))

#CUDA implementation
CUDA_MODULES_SRC := $(wildcard modules/impls_hw_accel/*.cu)
CUDA_MODULES := $(patsubst %.cu,%.o,$(CUDA_MODULES_SRC))
CUDA_MODULES_LINKED := $(patsubst %,%.linked.o,$(CUDA_MODULES_SRC))

LDFLAGS_CUDA := -I/opt/cuda/include/ -L/opt/cuda/lib
LDLIBS_CUDA := -lcuda -lcudart -lnvjpeg_static -lculibos -lcudart -lcudadevrt

#General arguments
LDFLAGS := -I modules/ -I include/lodepng/ -I LRs/ -I modules/hw_accel/
CXXFLAGS := $(LDFLAGS) $(MODULES) $(LRS) Program.o -g

#Compile with LodePNG implementation (link object files)
graphics-lode.out: HW_ACCEL = LODE_IMPL
graphics-lode.out: $(MODULES) $(MODULES_SHARED_CPP) $(LRS) $(LODE) Program.o
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CPP) $(LODE) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -o graphics-lode.out

#Compile with CUDA implementation
graphics-cuda.out: HW_ACCEL = CUDA_IMPL
graphics-cuda.out: $(MODULES) $(MODULES_SHARED_CUDA) $(LRS) $(CUDA_MODULES) Program.o
	nvcc $(LDFLAGS) -dlink -o cuda_modules_linked.o $(MODULES_SHARED_CUDA) $(CUDA_MODULES) $(LDLIBS_CUDA)
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CUDA) cuda_modules_linked.o $(CUDA_MODULES) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -o graphics-cuda.out

modules/impls_shared/%.cu.o: modules/impls_shared/%.cpp
	nvcc $(LDFLAGS) -x cu -rdc=true --debug --device-debug --cudart shared -o $@ -c $^

#Compile CUDA implementation (target that invokes if *.o with *.cu source is required by other targets)
%.o: %.cu
	nvcc $(LDFLAGS) -rdc=true --debug --device-debug --cudart shared -o $@ -c $^

#Target that invokes if *.o file with *.cpp source is required by other targets
%.o: %.cpp
	$(CXX) $(LDFLAGS) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -g -o $@ -c $^

#Clean build files
clean:
	rm -f $(MODULES) $(MODULES_SHARED_CUDA) $(MODULES_SHARED_CUDA_LINKED) $(LRS) $(LODE) $(CUDA_MODULES) $(CUDA_MODULES_LINKED) Program.o graphics-lode.out graphics-cuda.out

#Clean program output files
clean-output-LR1:
	rm -f $(wildcard output/LR1/*)
clean-output-LR2:
	rm -f $(wildcard output/LR2/*)
