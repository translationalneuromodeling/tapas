# aponteeduardo@gmail.com
# copyright (C) 2014

# ===========================================================================
# Set up
# ===========================================================================

.PHONY = directories

MKDIR = mkdir -p
SRCDIR = ./
BUILDDIR = ./../build
LIBDIR = $(abs_top_builddir)/lib
BINDIR = ./../bin

# Other things

BDDM_MATLAB = @matlabhome@
# nvcc flags

NVCC = nvcc

LCUDA = -arch=sm_20 -L$(CUDA_LDFLAGS)/lib64 -lcuda -lcudart
LNUM = -lm

NVCCFLAGS = -arch=sm_20 --shared -Xcompiler -fPIC -c

# Mex Flags

MATFLAGS = -I$(BDDM_MATLAB)/extern/include -DMATLAB_MEX_FILE -std=c99 -D_GNU_SOURCE  -fexceptions -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O -DNDEBUG

LOP0 =-O -pthread -shared -Wl,--version-script,$(BDDM_MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined

LOP1 = -Wl,-rpath-link,/usr/local/MATLAB/R2012b/bin/glnxa64 -L$(BDDM_MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lstdc++ -L$(CUDA_LDFLAGS)/lib64 -lcudart 

# Rules

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# ===========================================================================
# Compilation 
# ===========================================================================

all: directories $(BINDIR)/c_mpdcm_fmri_euler.mexa64 \
	$(BINDIR)/c_mpdcm_fmri_kr4.mexa64 $(BINDIR)/c_mpdcm_num_devices.mexa64 \
	$(BINDIR)/c_mpdcm_set_device.mexa64 $(BINDIR)/c_mpdcm_get_device.mexa64

# ===========================================================================
# Directories
# ===========================================================================

directories: $(BUILDDIR) $(LIBDIR) $(BINDIR)

$(BUILDDIR): 
	$(MKDIR) $(BUILDDIR)
$(LIBDIR):
	$(MKDIR) $(LIBDIR)
$(BINDIR):
	$(MKDIR) $(BINDIR)

# ===========================================================================
# Link
# ===========================================================================

$(BINDIR)/c_mpdcm_fmri_euler.mexa64 : $(BUILDDIR)/c_mpdcm.o \
	$(LIBDIR)/libmpdcm.so $(BUILDDIR)/c_mpdcm_fmri_euler.o 

$(BINDIR)/c_mpdcm_fmri_kr4.mexa64 : $(BUILDDIR)/c_mpdcm.o \
	$(LIBDIR)/libmpdcm.so $(BUILDDIR)/c_mpdcm_fmri_kr4.o 

$(BINDIR)/c_mpdcm_num_devices.mexa64 : $(BUILDDIR)/c_mpdcm_num_devices.o
$(BINDIR)/c_mpdcm_get_device.mexa64 : $(BUILDDIR)/c_mpdcm_get_device.o
$(BINDIR)/c_mpdcm_set_device.mexa64 : $(BUILDDIR)/c_mpdcm_set_device.o

$(BINDIR)/%.mexa64 : $(BUILDDIR)/%.o
	$(CC) $(LOP0) -o $@ $^ $(LOP1)

# ===========================================================================
# Compile
# ===========================================================================

$(BUILDDIR)/c_mpdcm_fmri_euler.o $(BUILDDIR)/c_mpdcm_fmri_kr4.o :
	$(CC) -c $(MATFLAGS) -o $@ $^

$(BUILDDIR)/c_mpdcm_fmri_euler.o : $(SRCDIR)/c_mpdcm_fmri_euler.c
$(BUILDDIR)/c_mpdcm_fmri_kr4.o : $(SRCDIR)/c_mpdcm_fmri_kr4.c

# Device code

$(BUILDDIR)/c_mpdcm_num_devices.o : $(SRCDIR)/c_mpdcm_num_devices.c
$(BUILDDIR)/c_mpdcm_get_device.o : $(SRCDIR)/c_mpdcm_get_device.c
$(BUILDDIR)/c_mpdcm_set_device.o : $(SRCDIR)/c_mpdcm_set_device.c

$(BUILDDIR)/c_mpdcm_num_devices.o $(BUILDDIR)/c_mpdcm_get_device.o \
	$(BUILDDIR)/c_mpdcm_set_device.o :
	$(CC) -c $(MATFLAGS) $(CUDA_CFLAGS) -o $@ $<

# Compile c

$(BUILDDIR)/c_mpdcm.o : $(SRCDIR)/c_mpdcm.c
	$(CC) -c $(MATFLAGS) -o $@ $^

# Compile cuda

$(LIBDIR)/libmpdcm.so : $(BUILDDIR)/mpdcm.o
	$(NVCC) -shared -o $@ $^ $(LCUDA) 

$(BUILDDIR)/mpdcm.o : $(SRCDIR)/mpdcm.cu

clean :
	$(RM) -r $(BINDIR)/*.mexa64
	$(RM) -r $(BUILDDIR)/*.o
	$(RM) -r $(LIBDIR)/*.so 
