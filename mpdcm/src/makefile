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

BDDM_MATLAB = $(shell matlab -nodisplay -r "fprintf(1, matlabroot); fprintf(1, '\n'); exit;" | tail -n 2| head -n 1)

# nvcc flags

NVCC = nvcc
MATCC = gcc

ICUDA = -I$(CUDA_PATH)/include

LCUDA = -arch=sm_20 -L$(CUDA_PATH)/lib64 -lcuda -lcudart
LNUM = -lm

NVCCFLAGS = -arch=sm_20 --shared -Xcompiler -fPIC -c

# Mex Flags

MATFLAGS = -I$(BDDM_MATLAB)/extern/include -DMATLAB_MEX_FILE -std=c99 -D_GNU_SOURCE  -fexceptions -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O -DNDEBUG

LOP0 =-O -pthread -shared -Wl,--version-script,$(BDDM_MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined

LOP1 = -Wl,-rpath-link,/usr/local/MATLAB/R2012b/bin/glnxa64 -L$(BDDM_MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lstdc++ -L$(CUDA_PATH)/lib64 -lcudart 

# Rules

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# ===========================================================================
# Compilation 
# ===========================================================================

all: directories c_mpdcm_fmri_euler.mexa64 c_mpdcm_fmri_kr4.mexa64 c_mpdcm_num_devices.mexa64 c_mpdcm_set_device.mexa64 c_mpdcm_get_device.mexa64

# Directories

directories: $(OUTDIR)

$(OUTDIR): 
	$(MKDIR) $(OUTDIR)

# Link

c_mpdcm_fmri_euler.mexa64 : c_mpdcm.o libmpdcm.so c_mpdcm_fmri_euler.o 
c_mpdcm_fmri_kr4.mexa64 : c_mpdcm.o libmpdcm.so c_mpdcm_fmri_kr4.o 
c_mpdcm_num_devices.mexa64 : c_mpdcm_num_devices.o
c_mpdcm_get_device.mexa64 : c_mpdcm_get_device.o
c_mpdcm_set_device.mexa64 : c_mpdcm_set_device.o

%.mexa64 : %.o
	gcc $(LOP0) -o $@ $< $(LOP1)

# Compile mex

c_mpdcm_fmri_euler.o : c_mpdcm_fmri_euler.c
c_mpdcm_fmri_kr4.o : c_mpdcm_fmri_kr4.c


c_mpdcm_fmri_euler.o c_mpdcm_fmri_kr4.o :
	$(MATCC) -c $(MATFLAGS) -o $@ $^


# Device code

c_mpdcm_num_devices.o : c_mpdcm_num_devices.c
c_mpdcm_get_device.o : c_mpdcm_get_device.c
c_mpdcm_set_device.o : c_mpdcm_set_device.c

c_mpdcm_num_devices.o c_mpdcm_get_device.o c_mpdcm_set_device.o :
	$(MATCC) -c $(MATFLAGS) $(ICUDA) -o $@ $<

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
