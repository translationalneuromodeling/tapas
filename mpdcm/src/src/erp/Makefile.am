
TARGETS = erp
LIBDIR = $(abs_top_srcdir)/lib
INCDIR = -I$(abs_top_srcdir)/src/includes -I$(abs_top_srcdir)
LIBDCM = $(LIBDIR)/lib$(TARGETS).a
SOURCES = $(filter-out %.aux.cu, $(wildcard *.cu))

AUXFILES = $(addsuffix .aux.cu, $(TARGETS))
AUXOBJS = $(AUXFILES:%.cu=%.o)

NVCCFLAGS = -arch=sm_20 -Xcompiler -fPIC

all: $(LIBDCM)

$(LIBDCM): $(LIBDIR)/lib%.a : %.aux.o | $(LIBDIR)
	$(NVCC) -lib $(NVCCFLAGS) $(INCDIR) -o $@ $^

$(AUXOBJS) : %.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCDIR) $^ -o $@
	
# Because it needs to be compatible with an old version of cuda.
# Wait to get $(AUXFILES) expanded so that it can be used for the second pass

.SECONDEXPANSION:
$(AUXFILES) : $$(filter $$(basename $$(basename $$(AUXFILES)))%, $$(SOURCES))
	cat $^ > $@

$(SOURCES) : %.cu : %.hcu

#$(SOURCES) : $(erp)_kernels.hcu

$(LIBDIR):
	$(MKDIR_P) $@

clean:
	$(RM) *.aux.cu
	$(RM) *.o
	$(RM) $(LIBDCM)

.PHONY : all clean
