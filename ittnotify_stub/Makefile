CC ?= cc
CFLAGS ?= -O2 -fPIC

all: libittnotify.so

libittnotify.so: libittnotify_stub.c
	$(CC) $(CFLAGS) -shared -o $@ $<

clean:
	rm -f libittnotify.so

.PHONY: all clean

