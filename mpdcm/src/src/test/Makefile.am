
BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -v -largeArrayDims -O GCC=$(CC)
MEXEXT = mexa64

TDIR = $(abs_top_srcdir)/test
IDIR = -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/includes -I$(abs_top_srcdir)
LDIR = $(abs_top_srcdir)/lib

SOURCES := $(wildcard *.cpp) 
OBJS = $(SOURCES:.cpp=.o)
mTESTS = $(addprefix $(TDIR)/, $(SOURCES:.cpp=.$(MEXEXT)))

LDFLAGS = -L$(LDIR) $(CUDA_LDFLAGS)
LDLIBS = -lmpdcm -lfmri $(LIBS) $(CUDA_LIBS)

all: $(mTESTS)

$(OBJS) : %.o : %.cpp
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ $(IDIR)  $(CUDA_CFLAGS) $^

$(mTESTS) : $(TDIR)/%.$(MEXEXT) : %.o | $(TDIR)
	$(MEXCC) $(MEXFLAGS) $(LDFLAGS) -outdir $(TDIR) $^ $(LDLIBS)

$(mTESTS) : | $(abs_top_srcdir)/lib/libmpdcm.a $(abs_top_srcdir)/lib/libfmri.a
 
$(TDIR):
	$(MKDIR_P) $@

clean:
	$(RM) *.o
	$(RM) $(mTESTS)

