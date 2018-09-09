# aponteeduardo@gmail.com
# copyright (C) 2015


BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -v -largeArrayDims -O GCC=$(CC)
MEXEXT = @mexext@

IDIR = -I$(abs_top_srcdir) -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/includes
LDIR = $(abs_top_srcdir)/lib
BINDIR = $(abs_top_srcdir)/bin

SOURCES := $(wildcard *.cpp)
OBJS = $(SOURCES:.cpp=.o)
MBINS = $(addprefix $(BINDIR)/,$(SOURCES:.cpp=.$(MEXEXT)))

LDFLAGS += -L$(LDIR) 

CXXFLAGS += -fPIC
CFLAGS += $(OPENMP_CFLAGS)

all: $(MBINS)

$(BINDIR):
	$(MKDIR_P) $@

$(OBJS) : %.o : %.cpp
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ $(IDIR) CXXFLAGS="$(CXXFLAGS) $(CFLAGS)" $^

$(MBINS) : $(BINDIR)/%.$(MEXEXT) : %.o | $(BINDIR)
	$(MEXCC) $(MEXFLAGS) $(LDFLAGS) -outdir $(BINDIR) $^ $(LIBS)

$(MBINS) : $(abs_top_srcdir)/lib/libmpdcmhost.a 

clean:
	$(RM) $(BINDIR)/*
	$(RM) *.o
