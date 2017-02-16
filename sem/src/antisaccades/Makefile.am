

MEXCC = @matlabhome@/bin/mex
# Choose with compiler to use
MEXFLAGS = -v -largeArrayDims -O CC=$(CC)

IDIR = $(abs_top_srcdir)/src
LIBDIR = $(abs_top_srcdir)/lib

LIBSOONER = $(LIBDIR)/libantisaccades.a
SOURCES = $(wildcard *.c)
OBJS = $(SOURCES:%.c=%.o)
CFLAGS += $(OPENMP_CFLAGS) -I$(IDIR) -std=c99 -DTAPAS_MATLAB -fPIC

all: $(LIBSOONER)

$(LIBSOONER): $(OBJS) | $(LIBDIR)
	ar rcs $@ $^

$(OBJS) : %.o : %.c
	$(MEXCC) $(MEXFLAGS) -c -outdir ./ -output $@ CFLAGS="$(CFLAGS)" $^
	
$(LIBDIR):
	$(MKDIR_P) $@

clean:
	$(RM) *.o
	$(RM) $(LIBSOONER)

