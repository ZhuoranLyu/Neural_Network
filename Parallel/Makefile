EXECUTABLES = NN
CC=gcc

all: $(EXECUTABLES)

LDFLAGS += $(foreach librarydir,$(subst :, ,$(LD_LIBRARY_PATH)),-L$(librarydir))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  LDFLAGS += -lrt -lOpenCL -lm
  CFLAGS += -Wall -std=gnu99 -g -O2
endif
ifeq ($(UNAME_S),Darwin)
  LDFLAGS +=  -framework OpenCL -lm
  CFLAGS += -Wall -std=c99 -g -O2
endif

ifdef OPENCL_INC
  CPPFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  LDFLAGS = -L$(OPENCL_LIB)
endif


host.o: host.c cl-helper.h util.h helper.h
cl-helper.o: cl-helper.c cl-helper.h
helper.o: helper.c helper.h

NN: host.o cl-helper.o helper.o
	${CC} -o ${EXECUTABLES} host.o cl-helper.o helper.o ${LDFLAGS} ${CFLAGS}

clean:
	rm -f $(EXECUTABLES) *.o