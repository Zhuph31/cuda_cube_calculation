################################################################################
# CAUTION: MAKE SURE YOUR IMPLEMENTATION COMPILES WITH THE ORIGINAL MAKEFILE
################################################################################
CC=g++
NVCC=nvcc
CXXFLAGS= -Wall -std=c++17 -O2 -MD -MP -g
CUDAFLAGS= $(aflags) -std=c++17 -arch=compute_86 -MD -MP 
LIBS= -lcudart
LIBDIRS=
INCDIRS=
EXE=zhupeng8
OBJS=zhupeng8.o 
AUTODEPS=$(OBJS:.o=.d)

.PHONY: all clean
all: $(EXE)

$(EXE): $(OBJS)
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) $(LIBDIRS) $^ $(LIBS) -o $(EXE)

clean:
	rm -f *.d *.o $(EXE)

%.o: %.cu
	$(NVCC) -dc $(CUDAFLAGS) $(INCDIRS) $< -o $@

%.o: %.cpp
	$(CC) -c $(CXXFLAGS) $(INCDIRS) $< -o $@

-include $(AUTODEPS)