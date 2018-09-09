
BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -v -largeArrayDims -O GCC=$(CC)
MEXEXT = mexa64
NVCCFLAGS = -arch=sm_20 -Xcompiler -fPIC

SRCDIR = $(abs_top_srcdir)
LIBDIR = $(SRCDIR)/lib

LIBMPDCM = $(LIBDIR)/libmpdcm.a

LOCINC = -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/includes -I$(abs_top_srcdir) 

all: $(LIBDIR) $(LIBMPDCM)

SOURCES := $(wildcard *.cpp)
OBJS = $(SOURCES:.cpp=.o)

$(LIBMPDCM) : $(OBJS)
	$(NVCC) --lib $(NVCCFLAGS) -o $@ $^
#	ar rvs $@ $^

$(OBJS) : %.o : %.cpp | $(LIBDIR)
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ $(CUDA_CFLAGS) $(LOCINC) CXXFLAGS="\$$CXXFLAGS" $^

$(LIBDIR):
	$(MKDIR_P) $@

clean:
	$(RM) $(OBJS)
	$(RM) $(LIBMPDCM)

.PHONY : all clean

