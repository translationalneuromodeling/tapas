# aponteeduardo@gmail.com
# copyright (C) 2015


BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -v -largeArrayDims -O CC=$(CC) CXX=$(CXX)
MEXEXT = @mexext@

IDIR = -I$(abs_top_srcdir)/antisaccades -I$(abs_top_srcdir)
LDIR = $(abs_top_srcdir)/lib
BINDIR = $(abs_top_srcdir)/bin

UTILS := mexutils.c mexsummaries.c
UTILS_OBJS = $(UTILS:.c=.o)
SOURCES := $(wildcard c_*.c)
OBJS = $(SOURCES:.c=.o)
MBINS = $(addprefix $(BINDIR)/,$(SOURCES:.c=.$(MEXEXT)))

LDFLAGS += -L$(LDIR)
CFLAGS += $(OPENMP_CFLAGS) -std=c11 -fPIC $(IDIR)

all: $(MBINS)

$(BINDIR):
	$(MKDIR_P) $@

$(OBJS) : %.o : %.c
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ CFLAGS="$(CFLAGS)" $^

$(UTILS_OBJS) : %.o : %.c
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ CFLAGS="$(CFLAGS)" $^

$(MBINS) : $(BINDIR)/%.$(MEXEXT) : %.o | $(BINDIR) 
	$(MEXCC) $(MEXFLAGS) $(LDFLAGS) CFLAGS="$(CFLAGS)" -outdir $(BINDIR) $^ $(LIBS) -lantisaccades

$(MBINS) : $(abs_top_srcdir)/lib/libantisaccades.a
$(MBINS) : $(UTILS_OBJS)

clean:
	$(RM) $(BINDIR)/*
	$(RM) *.o
