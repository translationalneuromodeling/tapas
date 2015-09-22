# aponteeduardo@gmail.com
# copyright (C) 2014

# ===========================================================================
# Set up
# ===========================================================================

.PHONY = $(MDIRS)

#NVCCFLAGS = -arch=sm_20 -rdc=true -Xcompiler -fPIC

NVCCFLAGS = -arch=sm_20 -Xcompiler -fPIC

# Mex Flags

BDDM_MATLAB = @matlabhome@

MEXCC = $(BDDM_MATLAB)/bin/mex
MEXFLAGS = -silent -largeArrayDims -O
MEXEXT = mexa64

LOP1 = $(CUDA_LDFLAGS) $(LIBS) 

# ===========================================================================
# Compilation 
# ===========================================================================

SRCDIR = ./
BUILDDIR = ./../build
LIBDIR = $(abs_top_builddir)/lib
BINDIR = ./../../matlab/

MDIRS = $(BUILDDIR) $(LIBDIR) $(BINDIR)

MEXDCMBIN = $(BINDIR)/c_mpdcm_fmri_euler.$(MEXEXT) \
	$(BINDIR)/c_mpdcm_fmri_kr4.$(MEXEXT) $(BINDIR)/c_mpdcm_fmri_bs.$(MEXEXT) 

MEXDEVBIN = $(BINDIR)/c_mpdcm_num_devices.$(MEXEXT) \
	$(BINDIR)/c_mpdcm_get_device.$(MEXEXT) \
	$(BINDIR)/c_mpdcm_set_device.$(MEXEXT)

all: $(MDIRS) $(MEXDCMBIN) $(MEXDEVBIN)

# ===========================================================================
# Directories
# ===========================================================================

$(MDIRS):
	$(MKDIR_P) $(MDIRS)

# ===========================================================================
# Link
# ===========================================================================

$(MEXDCMBIN) : $(BINDIR)/%.$(MEXEXT) : $(BUILDDIR)/%.o $(LIBDIR)/libmpdcm.a \
		$(BUILDDIR)/c_mpdcm.o
	$(MEXCC) $(MEXFLAGS) -outdir $(BINDIR) $^ $(LOP1)

$(MEXDEVBIN) : $(BINDIR)/%.$(MEXEXT) : $(BUILDDIR)/%.o
	$(MEXCC) $(MEXFLAGS) -outdir $(BINDIR) $^ $(LOP1)

# Again for compatility with cuda 4.2

# This would be the case if compiling from different files.

#$(BUILDDIR)/devlink.o : $(CUDAOBJ)  
#	$(NVCC) --device-link $(NVCCFLAGS) $^ -o $@
#
#$(LIBDIR)/libmpdcm.a : $(CUDAOBJ) $(BUILDDIR)/devlink.o
#	$(NVCC) --lib $(NVCCFLAGS) -o $@ $^

# Instead directly compile the library.

$(LIBDIR)/libmpdcm.a : $(CUDAOBJ)
	$(NVCC) --lib $(NVCCFLAGS) -o $@ $^

# ===========================================================================
# Compile
# ===========================================================================

# Compile mex

MEXDCMOBJ = $(BUILDDIR)/c_mpdcm_fmri_euler.o $(BUILDDIR)/c_mpdcm_fmri_kr4.o \
	$(BUILDDIR)/c_mpdcm_fmri_bs.o $(BUILDDIR)/c_mpdcm.o 

# Device code

MEXDEVOBJ = $(BUILDDIR)/c_mpdcm_num_devices.o $(BUILDDIR)/c_mpdcm_get_device.o \
	$(BUILDDIR)/c_mpdcm_set_device.o 	 

$(MEXDEVOBJ) $(MEXDCMOBJ) : $(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(MEXCC) $(MEXFLAGS) -c -outdir $(BUILDDIR) -output $@ CFLAGS="\$$CFLAGS -std=c99 $(CUDA_CFLAGS)" $^

# Compile cuda

AUXOBJ = $(BUILDDIR)/mpdcm.o $(BUILDDIR)/kernels.o $(BUILDDIR)/dynamics.o \
	$(BUILDDIR)/updates.o $(BUILDDIR)/integrators.o

SRCCUDA = $(SRCDIR)/mpdcm.cu $(SRCDIR)/kernels.cu $(SRCDIR)/dynamics.cu \
	$(SRCDIR)/updates.cu $(SRCDIR)/integrators.cu

CUDAOBJ = $(BUILDDIR)/aux_mpdcm.o

$(CUDAOBJ) : $(BUILDDIR)/%.o : $(SRCDIR)/%.cu 
	$(NVCC) -c $(NVCCFLAGS) $(DEFS) $^ -o $@

$(SRCDIR)/aux_mpdcm.cu : $(SRCCUDA)
	cat $^ > $@


clean :
	$(RM) $(BINDIR)/*.$(MEXEXT)
	$(RM) $(BUILDDIR)/*.o
	$(RM) $(LIBDIR)/*.so
	$(RM) $(SRCDIR)/aux_mpdcm.cu
