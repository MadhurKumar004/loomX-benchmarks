SHELL=/bin/sh

CLASS=S
BENCHMARK=ep
BENCHMARKU=EP
PARAMETERS=setparams

include ../config/make.def

ifeq ($(HIP_PLATFORM),amd)
	CFLAGS = $(AMD_FLAGS)
else ifeq ($(HIP_PLATFORM),nvidia)
	CFLAGS = $(NVIDIA_FLAGS)
else ifeq ($(HIP_PLATFORM),intel)
	CFLAGS = $(INTEL_FLAGS)
	CC = $(CC_INTEL)
else
    CFLAGS = $(AMD_FLAGS)
endif

COMPILE_EP=$(CC) $(CFLAGS) $(BENCHMARK).cpp -DCLASS_$(CLASS) -o $(BENCHMARK).$(CLASS)
SET_PARAMETERS=$(CC) $(SYSDIR)/$(PARAMETERS).cpp -o $(PARAMETERS) && ./$(PARAMETERS) $(BENCHMARK) $(CLASS)
MOVE_BINARIES=mv $(BENCHMARK).$(CLASS) $(BINDIR) && mv $(PARAMETERS) $(BINDIR)

$(BENCHMARK):
	$(pwd) $(SET_PARAMETERS) && $(COMPILE_EP) && $(MOVE_BINARIES)

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.hpp core
