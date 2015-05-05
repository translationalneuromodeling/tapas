# aponteeduardo@gmail.com
# copyright (C) 2014

# ===========================================================================
# Set up
# ===========================================================================

.PHONY = directories


NVCCFLAGS = -arch=sm_20 -rdc=true -Xcompiler -fPIC

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
BINDIR = ./../bin

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
	$(MKDIR_P) $^

# ===========================================================================
# Link
# ===========================================================================

$(MEXDCMBIN) : $(BINDIR)/%.$(MEXEXT) : $(BUILDDIR)/%.o $(LIBDIR)/libmpdcm.a \
		$(BUILDDIR)/c_mpdcm.o
	$(MEXCC) $(MEXFLAGS) -outdir $(BINDIR) $^ $(LOP1)

$(MEXDEVBIN) : $(BINDIR)/%.$(MEXEXT) : $(BUILDDIR)/%.o
	$(MEXCC) $(MEXFLAGS) -outdir $(BINDIR) $^ $(LOP1)

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
	$(MEXCC) $(MEXFLAGS) -c -outdir $(BUILDDIR) $@ CFLAGS="\$$CFLAGS -std=c99 $(CUDA_CFLAGS)" $^

# Compile cuda

CUDAOBJ = $(BUILDDIR)/mpdcm.o $(BUILDDIR)/kernels.o $(BUILDDIR)/dynamics.o \
	$(BUILDDIR)/updates.o $(BUILDDIR)/integrators.o

$(CUDAOBJ) : $(BUILDDIR)/%.o : $(SRCDIR)/%.cu 
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/devlink.o : $(CUDAOBJ)  
	$(NVCC) --device-link $(NVCCFLAGS) $^ -o $@

$(LIBDIR)/libmpdcm.a : $(CUDAOBJ) $(BUILDDIR)/devlink.o
	$(NVCC) --lib $(NVCCFLAGS) -o $@ $^

clean :
	$(RM) -r $(BINDIR)/*.$(MEXEXT)
	$(RM) -r $(BUILDDIR)/*.o
	$(RM) -r $(LIBDIR)/*.so 
