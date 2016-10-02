LIBDIR=$(HOME)/Research/dev/lib/
LIBS=-lgsl -lconfig++ -lergopack -lgomp
CONFIG=../cfg/readConfig

CC=g++
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -pedantic-errors -O3
LDFLAGSSTD=-undefined dynamic_lookup
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

EXE=transfer

all:$(EXE).out
	mkdir -p ../results/transfer/forwardTransition
	mkdir -p ../results/transfer/backwardTransition
	mkdir -p ../results/transfer/initDist
	mkdir -p ../results/transfer/finalDist
	mkdir -p ../results/transfer/gridMem

$(EXE).o:$(EXE).cpp
	$(CC) -c $(CFLAGS) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) -c $(CFLAGS) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o
	$(CC) $(LDFLAGS) -o $@ $^ -L$(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o $(EXE).out
