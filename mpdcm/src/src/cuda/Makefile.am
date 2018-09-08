
TARGETS = fmri
INCDIR = -I$(abs_top_srcdir)/src/includes -I$(abs_top_srcdir)
LIBDIR = $(abs_top_srcdir)/lib
LIBDCM = $(LIBDIR)/libfmri.a
SOURCES = $(filter-out %.aux.cu, $(wildcard *.cu))

AUXFILES = $(addsuffix .aux.cu, $(TARGETS))
AUXOBJS = $(AUXFILES:%.cu=%.o)

NVCCFLAGS = -arch=sm_20 -Xcompiler -fPIC

all: $(LIBDCM)

$(LIBDCM): $(LIBDIR)/lib%.a : %.aux.o | $(LIBDIR)
	$(NVCC) $(INCDIR) -lib $(NVCCFLAGS) -o $@ $^

$(AUXOBJS) : %.o : %.cu
	$(NVCC) $(INCDIR) -c $(NVCCFLAGS) $^ -o $@
	
# Because it needs to be compatible with an old version of cuda.
# Wait to get $(AUXFILES) expanded so that it can be used for the second pass

.SECONDEXPANSION:
$(AUXFILES) : $$(filter $$(basename $$(basename $$(AUXFILES)))%, $$(SOURCES))
	cat $^ > $@

$(SOURCES) : %.cu : %.hcu

$(SOURCES) : fmri_kernels.hcu

$(LIBDIR):
	$(MKDIR_P) $@

clean:
	$(RM) *.aux.cu
	$(RM) *.o
	$(RM) $(LIBDCM)

.PHONY : all clean
