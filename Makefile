.PHONY: clean clean-output-LR1 clean-output-LR2

#Main compiler
CXX = g++

OPTIMIZATION_FLAGS_RELEASE= -march=native -Ofast
OPTIMIZATION_FLAGS_DEBUG= -O0 -g
OPTIMIZATION_FLAGS=$(OPTIMIZATION_FLAGS_RELEASE)

NV_OPTIMIZATION_FLAGS_RELEASE= -use_fast_math -v
NV_OPTIMIZATION_FLAGS_DEBUG= --debug --device-debug --cudart shared
NV_OPTIMIZATION_FLAGS= $(NV_OPTIMIZATION_FLAGS_RELEASE)

QT_DIR=/usr/lib/qt6

#GUI
GUI_SRC=gui/mainwindow.cpp gui/moc_mainwindow.cpp
GUI=$(patsubst %.cpp,%.o,$(GUI_SRC))

LDFLAGS_GUI=-I/usr/include/qt6 -I/usr/include/qt6/QtGui -I/usr/include/qt6/QtCore -I/usr/include/qt6/QtWidgets -I/usr/lib/qt6/mkspecs/linux-g++ -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -fPIC
LD_LIBS_GUI=-lQt6Core -lQt6Gui -lQt6Widgets -ltbb

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
LDLIBS_CUDA := -lcuda -lcudart -lnvjpeg_static -lculibos -lcudart -lcudadevrt -lcublas

#General arguments
LDFLAGS := -I modules/ -I include/lodepng/ -I LRs/
CXXFLAGS := -std=c++17 $(LDFLAGS) $(LDFLAGS_GUI) $(MODULES) $(LRS) $(GUI) -pthread Program.o -g

#Compile with LodePNG implementation (link object files)
graphics-lode.out: HW_ACCEL = LODE_IMPL
graphics-lode.out: $(MODULES) $(MODULES_SHARED_CPP) $(LRS) $(LODE) $(GUI) Program.o
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CPP) $(LODE) $(LD_LIBS_GUI) -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o graphics-lode.out

#Compile with CUDA implementation
graphics-cuda.out: HW_ACCEL = CUDA_IMPL
graphics-cuda.out: $(MODULES) $(MODULES_SHARED_CUDA) $(LRS) $(CUDA_MODULES) $(GUI) Program.o
	nvcc $(LDFLAGS) -arch=native -dlink -o cuda_modules_linked.o $(MODULES_SHARED_CUDA) $(CUDA_MODULES) $(LDLIBS_CUDA)
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CUDA) cuda_modules_linked.o $(CUDA_MODULES) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) $(LD_LIBS_GUI) -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o graphics-cuda.out

modules/impls_shared/%.cu.o: modules/impls_shared/%.cpp
	nvcc $(LDFLAGS) -arch=native -x cu -rdc=true $(NV_OPTIMIZATION_FLAGS) -o $@ -c $^

#Compile CUDA implementation (target that invokes if *.o with *.cu source is required by other targets)
%.o: %.cu
	nvcc $(LDFLAGS) -arch=native -rdc=true $(NV_OPTIMIZATION_FLAGS) -o $@ -c $^

gui/moc_mainwindow.cpp: gui/mainwindow.h gui/ui_mainwindow.h
	$(QT_DIR)/moc $(LDFLAGS) $< -o $@

gui/ui_mainwindow.h: gui/mainwindow.ui
	$(QT_DIR)/uic gui/mainwindow.ui -o gui/ui_mainwindow.h 


gui/mainwindow.o: gui/ui_mainwindow.h

#Target that invokes if *.o file with *.cpp source is required by other targets
%.o: %.cpp
	$(CXX) -std=c++17 $(LDFLAGS) $(LDFLAGS_CUDA) $(LDFLAGS_GUI) $(LDLIBS_CUDA) $(LD_LIBS_GUI) -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o $@ -c $< 

#Clean build files
clean:
	rm -f $(MODULES) $(MODULES_SHARED_CUDA) $(MODULES_SHARED_CUDA_LINKED) $(LRS) $(LODE) $(CUDA_MODULES) $(CUDA_MODULES_LINKED) gui/mainwindow.o gui/moc_mainwindow.cpp gui/ui_mainwindow.h Program.o graphics-lode.out graphics-cuda.out

#Clean program output files
clean-output-LR1:
	rm -f $(wildcard output/LR1/*)
clean-output-LR2:
	rm -f $(wildcard output/LR2/*)
