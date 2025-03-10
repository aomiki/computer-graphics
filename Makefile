.PHONY: clean clean-output-LR1 clean-output-LR2

#Main compiler
CXX = g++

#Modules
MODULES_SRC = $(wildcard modules/impls/*.cpp)
MODULES := $(patsubst %.cpp,%.o,$(MODULES_SRC))

#Laboratory works
LRS_SRC := $(wildcard LRs/impls/*.cpp)
LRS := $(patsubst %.cpp,%.o,$(LRS_SRC))

#LodePNG implementation
LODE_SRC := $(wildcard include/lodepng/lodepng.cpp modules/impls_cpu/*.cpp)
LODE :=  $(patsubst %.cpp,%.o,$(LODE_SRC))

#CUDA implementation
CUDA_MODULES_SRC := $(wildcard modules/impls_hw_accel/*.cu)
CUDA_MODULES := $(patsubst %.cu,%.o,$(CUDA_MODULES_SRC))
CUDA_MODULES_LINKED := $(patsubst %,%_link.o,$(CUDA_MODULES))

LDFLAGS_CUDA := -I/opt/cuda/include/ -L/opt/cuda/lib
LDLIBS_CUDA := -lcuda -lcudart -lnvjpeg_static -lculibos -lcudart -lcudadevrt

#General arguments
LDFLAGS := -I modules/ -I include/lodepng/ -I LRs/ -I modules/hw_accel/
CXXFLAGS := $(LDFLAGS) $(MODULES) $(LRS) Program.o -g

#Compile with LodePNG implementation (link object files)
graphics-lode.out: HW_ACCEL = LODE_IMPL
graphics-lode.out: $(MODULES) $(LRS) $(LODE) Program.o
	$(CXX) $(CXXFLAGS) $(LODE) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -o graphics-lode.out

#Compile with CUDA implementation
graphics-cuda.out: HW_ACCEL = CUDA_IMPL
graphics-cuda.out: $(MODULES) $(LRS) $(CUDA_MODULES) Program.o
	$(CXX) $(CXXFLAGS) $(CUDA_MODULES) $(CUDA_MODULES_LINKED) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -o graphics-cuda.out

#Compile CUDA implementation (target that invokes if *.o with *.cu source is required by other targets)
%.o: %.cu
	nvcc $(LDFLAGS) -rdc=true --debug  --device-debug --cudart shared -o $@ -c $^
	nvcc $(LDFLAGS) -dlink -o $@_link.o $@ $(LDLIBS_CUDA)

#Target that invokes if *.o file with *.cpp source is required by other targets
%.o: %.cpp
	$(CXX) $(LDFLAGS) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -g -o $@ -c $^

#Clean build files
clean:
	rm -f $(MODULES) $(LRS) $(LODE) $(CUDA_MODULES) Program.o graphics-lode.out graphics-cuda.out

#Clean program output files
clean-output-LR1:
	rm -f $(wildcard output/LR1/*)
clean-output-LR2:
	rm -f $(wildcard output/LR2/*)
