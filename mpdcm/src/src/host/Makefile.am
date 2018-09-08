
BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -v -largeArrayDims -O GCC=$(CC)
MEXEXT = @mexext@

SRCDIR = $(abs_top_srcdir)
LIBDIR = $(SRCDIR)/lib

LIBMPDCM = $(LIBDIR)/libmpdcmhost.a

LOCINC = -I$(abs_top_srcdir) -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/includes  

all: $(LIBDIR) $(LIBMPDCM)

SOURCES := $(wildcard *.cpp)
OBJS = $(SOURCES:.cpp=.o)

$(LIBMPDCM) : $(OBJS)
	ar rvs $@ $^

CXXFLAGS += -fPIC
CFLAGS += $(OPENMP_CFLAGS)

$(OBJS) : %.o : %.cpp | $(LIBDIR)
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ $(LOCINC)  CXXFLAGS="$(CXXFLAGS) $(CFLAGS)"  $^

$(LIBDIR):
	$(MKDIR_P) $@

clean:
	$(RM) $(OBJS)
	$(RM) $(LIBMPDCM)

.PHONY : all clean

