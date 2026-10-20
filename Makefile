# Builds the firmware's safe-map C (core/safemap.c, the code the node runs) as the shared library
# that twin/bindings/core_ffi.py loads.
CC ?= cc

twin/libcore.so: core/safemap.c core/safemap.h
	$(CC) -std=c11 -O2 -fPIC -shared -Icore -o $@ core/safemap.c -lm

clean:
	rm -f twin/libcore.so

.PHONY: clean
