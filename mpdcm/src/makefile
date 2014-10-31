# aponteeduardo@gmail.com
# copyright (C) 2014

MAJOR := 0
MINOR := 1
VERSION := $(MAJOR).$(MINOR)

NAME := mpdcm



# ===========================================================================
# Set up
# ===========================================================================

.PHONY = directories

MKDIR = mkdir -p
OUTDIR = build

# Other things

BDDM_MATLAB = /usr/local/MATLAB/R2014a

# nvcc flags

NVCC = nvcc
MATCC = gcc

ICUDA = -I/usr/local/cuda-6.5/include

LCUDA = -arch=sm_30 -L/usr/local/cuda/lib64 -lcuda -lcudart
LNUM = -lm

NVCCFLAGS = -arch=sm_30 -rdc=true --shared -Xcompiler -fPIC -c

# Mex Flags

MATFLAGS = -I$(BDDM_MATLAB)/extern/include -DMATLAB_MEX_FILE -std=c99 -D_GNU_SOURCE  -fexceptions -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O -DNDEBUG

LOP0 =-O -pthread -shared -Wl,--version-script,$(BDDM_MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined

LOP1 = -Wl,-rpath-link,/usr/local/MATLAB/R2012b/bin/glnxa64 -L$(BDDM_MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lstdc++ -L/usr/local/cuda/lib64 -lcudart 

# Rules

%.mexa64 : %.o
	gcc $(LOP0) -o $@ $^ $(LOP1)

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# ===========================================================================
# Compilation 
# ===========================================================================

all: directories c_mpdcm_fmri_int.mexa64 c_mpdcm_num_devices.mexa64 c_mpdcm_set_device.mexa64 c_mpdcm_get_device.mexa64

# directories

directories: $(OUTDIR)

$(OUTDIR): 
	$(MKDIR) $(OUTDIR)

# Link

c_mpdcm_fmri_int.mexa64 : c_mpdcm.o libmpdcm.so c_mpdcm_fmri_int.o 
c_mpdcm_num_devices.mexa64 : c_mpdcm_num_devices.o
c_mpdcm_get_device.mexa64 : c_mpdcm_get_device.o
c_mpdcm_set_device.mexa64 : c_mpdcm_set_device.o


# Compile mex

c_mpdcm_fmri_int.o : c_mpdcm_fmri_int.c
	$(MATCC) -c $(MATFLAGS) -o $@ $^

c_mpdcm_num_devices.o : c_mpdcm_num_devices.c
	$(MATCC) -c $(MATFLAGS) $(ICUDA) -o $@ $^

c_mpdcm_get_device.o : c_mpdcm_get_device.c
	$(MATCC) -c $(MATFLAGS) $(ICUDA) -o $@ $^

c_mpdcm_set_device.o : c_mpdcm_set_device.c
	$(MATCC) -c $(MATFLAGS) $(ICUDA) -o $@ $^


# Compile c

c_mpdcm.o : c_mpdcm.c
	$(MATCC) -c $(MATFLAGS) -o $@ $^

# Compile cuda

mpdcm.o : mpdcm.cu

libmpdcm.so : mpdcm.o
	$(NVCC) -shared -o $@ $^ $(LCUDA) 


clean :
	$(RM) *.o *.so *.mexa64
	$(RM) -r $(OUTDIR) 
