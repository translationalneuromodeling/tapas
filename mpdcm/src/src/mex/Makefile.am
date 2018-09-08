# aponteeduardo@gmail.com
# copyright (C) 2015


BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -v -largeArrayDims -O GCC=$(CC)
MEXEXT = @mexext@

IDIR = -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/includes -I$(abs_top_srcdir)
LDIR = $(abs_top_srcdir)/lib
BINDIR = $(abs_top_srcdir)/bin

SOURCES := $(wildcard *.cpp)
OBJS = $(SOURCES:.cpp=.o)
MBINS = $(addprefix $(BINDIR)/,$(SOURCES:.cpp=.$(MEXEXT)))

LDFLAGS = -L$(LDIR) $(CUDA_LDFLAGS)
LDLIBS = -lmpdcm -lfmri $(LIBS) $(CUDA_LIBS)

all: $(MBINS)

$(BINDIR):
	$(MKDIR_P) $@

$(OBJS) : %.o : %.cpp
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@  $(CUDA_CFLAGS)  $(IDIR) $^

$(MBINS) : $(BINDIR)/%.$(MEXEXT) : %.o | $(BINDIR)
	$(MEXCC) $(MEXFLAGS) $(LDFLAGS) -outdir $(BINDIR) $^ $(LDLIBS)

$(MBINS) : $(abs_top_srcdir)/lib/libmpdcm.a $(abs_top_srcdir)/lib/libfmri.a 

clean:
	$(RM) $(BINDIR)/*
	$(RM) *.o
