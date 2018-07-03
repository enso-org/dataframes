CC = gcc
# CFLAGS: -fPIC -g
LDFLAGS = -shared -lstdc++ -dynamiclib

UNAME_S := $(shell uname -s)

SRCDIR=./native_libs/src
LIBDIR :=
ifeq ($(UNAME_S),Darwin)
	LIBDIR += ./native_libs/macos
else
	LIBDIR += ./native_libs/linux
endif

target1_SRC = $(SRCDIR)/csv.c
target2_SRC = $(SRCDIR)/2d_array.c
target3_SRC = $(SRCDIR)/utils.c
ALL_SRCS = $(target1_SRC) $(target2_SRC) $(target3_SRC)

TGT1 :=
TGT2 :=
TGT3 :=

ifeq ($(UNAME_S),Darwin)
	TGT1 += $(LIBDIR)/csv.dylib
	TGT2 += $(LIBDIR)/2d_array.dylib
	TGT3 += $(LIBDIR)/utils.dylib
else
	TGT1 += $(LIBDIR)/csv.so
	TGT2 += $(LIBDIR)/2d_array.so
	TGT3 += $(LIBDIR)/utils.so
endif


ALL_TGTS = $(TGT1) $(TGT2) $(TGT3)

csv: $(target1_SRC)
	$(CC) $(LDFLAGS) -o $(TGT1) $(target1_SRC)

arr2d: $(target2_SRC)
	$(CC) $(LDFLAGS) -o $(TGT2) $(target2_SRC)

utils: $(target3_SRC)
	$(CC) $(LDFLAGS) -o $(TGT3) $(target3_SRC)

all: csv arr2d utils


.PHONY: clean

clean:
	rm -rf $(ALL_TGTS)
