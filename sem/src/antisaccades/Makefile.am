#aponteeduardo@gmail.com

MEXCC = @matlabhome@/bin/mex
# Choose with compiler to use
MEXFLAGS = -v -largeArrayDims -O CC=$(CC)

IDIR = -I$(abs_top_srcdir)
LIBDIR = $(abs_top_srcdir)/lib

LIBSOONER = $(LIBDIR)/libantisaccades.a
SOURCES = $(wildcard *.c)
OBJS = $(SOURCES:%.c=%.o)
CFLAGS += $(OPENMP_CFLAGS) $(IDIR) -std=c11 -fPIC

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

