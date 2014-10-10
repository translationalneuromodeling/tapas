# Other things

BDDM_MATLAB = /usr/local/MATLAB/R2014a

# nvcc flags

NVCC = nvcc
MATCC = gcc


LCUDA = -arch=sm_30 -L/usr/local/cuda/lib64 -lcuda -lcudart
LNUM = -lm

NVCCFLAGS = -arch=sm_30 -rdc=true --shared -Xcompiler -fPIC -c

# Mex Flags

MATFLAGS = -I$(BDDM_MATLAB)/extern/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE  -fexceptions -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O -DNDEBUG

LOP0 =-O -pthread -shared -Wl,--version-script,$(BDDM_MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined

LOP1 = -Wl,-rpath-link,/usr/local/MATLAB/R2012b/bin/glnxa64 -L$(BDDM_MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lstdc++

# Rules

%.mexa64 : %.o
	gcc $(LOP0) -o $@ $^ $(LOP1)

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# ===========================================================================
# Compilation 
# ===========================================================================

all: c_mpdcm_fmri_int.mexa64

# Link

c_mpdcm_fmri_int.mexa64 : c_mpdcm.o mpdcm.so c_mpdcm_fmri_int.o 

# Compile mex

c_mpdcm_fmri_int.o : c_mpdcm_fmri_int.c
	$(MATCC) -c $(MATFLAGS) -o $@ $^

# Compile c

c_mpdcm.o : c_mpdcm.c
	$(MATCC) -c $(MATFLAGS) -o $@ $^

# Compile cuda

mpdcm.o : mpdcm.cu
mpdcm.so : mpdcm.o
	$(NVCC) -shared -o $@ $^ $(LCUDA) 


clean :
	$(RM) *.o *.so *.mexa64
