# aponteeduardo@gmail.com
# copyright (C) 2014

# ===========================================================================
# Set up
# ===========================================================================

.PHONY = directories

SRCDIR = ./
BUILDDIR = ./../build
LIBDIR = $(abs_top_builddir)/lib
BINDIR = ./../bin

# Other things

BDDM_MATLAB = @matlabhome@

LCUDA = -arch=sm_20 $(CUDA_LDFLAGS) -lcuda -lcudart
LNUM = -lm

NVCCFLAGS = -arch=sm_20 -rdc=true -Xcompiler -fPIC

# Mex Flags

MEXCC = $(BDDM_MATLAB)/bin/mex

MATFLAGS = -I$(BDDM_MATLAB)/extern/include -DMATLAB_MEX_FILE -std=c99 -D_GNU_SOURCE  -fexceptions -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O -DNDEBUG

LOP0 =-O -pthread -shared -Wl,--version-script,$(BDDM_MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined

LOP1 = $(CUDA_LDFLAGS) -lcudart 

# Rules

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# ===========================================================================
# Compilation 
# ===========================================================================

all: directories $(BINDIR)/c_mpdcm_fmri_euler.mexa64 \
	$(BINDIR)/c_mpdcm_fmri_kr4.mexa64 $(BINDIR)/c_mpdcm_fmri_bs.mexa64 \
	$(BINDIR)/c_mpdcm_num_devices.mexa64 $(BINDIR)/c_mpdcm_set_device.mexa64 \
	$(BINDIR)/c_mpdcm_get_device.mexa64

# ===========================================================================
# Directories
# ===========================================================================

directories: $(BUILDDIR) $(LIBDIR) $(BINDIR)

$(BUILDDIR): 
	$(MKDIR_P) $(BUILDDIR)
$(LIBDIR):
	$(MKDIR_P) $(LIBDIR)
$(BINDIR):
	$(MKDIR_P) $(BINDIR)

# ===========================================================================
# Link
# ===========================================================================

#$(BINDIR)/%.mexa64 : $(BUILDDIR)/%.o
#	$(CC) $(LOP0) -o $@ $^ $(LOP1)

$(BINDIR)/c_mpdcm_fmri_euler.mexa64 : $(BUILDDIR)/c_mpdcm.o \
	$(LIBDIR)/libmpdcm.a $(BUILDDIR)/c_mpdcm_fmri_euler.o 

$(BINDIR)/c_mpdcm_fmri_kr4.mexa64 : $(BUILDDIR)/c_mpdcm.o \
	$(LIBDIR)/libmpdcm.a $(BUILDDIR)/c_mpdcm_fmri_kr4.o 

$(BINDIR)/c_mpdcm_fmri_bs.mexa64 : $(BUILDDIR)/c_mpdcm.o \
	$(LIBDIR)/libmpdcm.a $(BUILDDIR)/c_mpdcm_fmri_bs.o 

$(BINDIR)/%.mexa64 : $(BUILDDIR)/%.o
	mex -v -n -outdir ./../bin/ -O $^ $(LOP1)

$(BINDIR)/c_mpdcm_num_devices.mexa64 : $(BUILDDIR)/c_mpdcm_num_devices.o
$(BINDIR)/c_mpdcm_get_device.mexa64 : $(BUILDDIR)/c_mpdcm_get_device.o
$(BINDIR)/c_mpdcm_set_device.mexa64 : $(BUILDDIR)/c_mpdcm_set_device.o

$(BINDIR)/%.mexa64 : $(BUILDDIR)/%.o
	mex -outdir ./../bin/ -output $@ -O $^ $(LOP1)
#	$(CC) $(LOP0) -o $@ $^ $(LOP1)

# ===========================================================================
# Compile
# ===========================================================================

MEXWRAPPERS = $(BUILDDIR)/c_mpdcm_fmri_euler.o $(BUILDDIR)/c_mpdcm_fmri_kr4.o \
	$(BUILDDIR)/c_mpdcm_fmri_bs.o $(BUILDDIR)/c_mpdcm.o 

$(MEXWRAPPERS):
	$(MEXCC) -c -outdir $(BUILDDIR) $@ -O CFLAGS="\$$CFLAGS -std=c99" $^

$(BUILDDIR)/c_mpdcm_fmri_euler.o : $(SRCDIR)/c_mpdcm_fmri_euler.c
$(BUILDDIR)/c_mpdcm_fmri_kr4.o : $(SRCDIR)/c_mpdcm_fmri_kr4.c
$(BUILDDIR)/c_mpdcm_fmri_bs.o : $(SRCDIR)/c_mpdcm_fmri_bs.c
$(BUILDDIR)/c_mpdcm.o : $(SRCDIR)/c_mpdcm.c

# Device code

$(BUILDDIR)/c_mpdcm_num_devices.o : $(SRCDIR)/c_mpdcm_num_devices.c
$(BUILDDIR)/c_mpdcm_get_device.o : $(SRCDIR)/c_mpdcm_get_device.c
$(BUILDDIR)/c_mpdcm_set_device.o : $(SRCDIR)/c_mpdcm_set_device.c

$(BUILDDIR)/c_mpdcm_num_devices.o $(BUILDDIR)/c_mpdcm_get_device.o \
	$(BUILDDIR)/c_mpdcm_set_device.o :
	$(MEXCC) -c -outdir $(BUILDDIR) $@ -O CFLAGS="\$$CFLAGS -std=c99 $(CUDA_CFLAGS)" $^

# Compile cuda

SRCCUDA = $(SRCDIR)/mpdcm.cu $(SRCDIR)/kernels.cu $(SRCDIR)/dynamics.cu \
	$(SRCDIR)/updates.cu $(SRCDIR)/integrators.cu

OBJCUDA = $(SRCCUDA:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

$(SRCDIR)/%.o : $(BUILDDIR)/%.cu
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/devlink.o : $(OBJCUDA)  
	$(NVCC) --device-link $(NVCCFLAGS) $^ -o $@

$(LIBDIR)/libmpdcm.a : $(OBJCUDA) $(BUILDDIR)/devlink.o
	$(NVCC) --lib $(NVCCFLAGS) -o $@ $^

clean :
	$(RM) -r $(BINDIR)/*.mexa64
	$(RM) -r $(BUILDDIR)/*.o
	$(RM) -r $(LIBDIR)/*.so 
