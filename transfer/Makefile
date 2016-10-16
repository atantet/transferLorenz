INCDIR=-I$(HOME)/local/include/
LIBDIR=-L$(HOME)/local/lib/
<<<<<<< HEAD
LIBS=-lgsl -lgslcblas -lconfig++ -lergopack -lgomp
=======
LIBS=-lgsl -lconfig++ -lergopack -lgomp
>>>>>>> fd01e4fd1720162ee0dde431bafdd1e8ec5d14c3
CONFIG=../cfg/readConfig

CC=icc -openmp
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -O3
LDFLAGSSTD=
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

EXE=transfer

all:$(EXE).out
	mkdir -p ../results/grid
	mkdir -p ../results/transfer/forwardTransition
	mkdir -p ../results/transfer/backwardTransition
	mkdir -p ../results/transfer/initDist
	mkdir -p ../results/transfer/finalDist
	mkdir -p ../results/transfer/gridMem
	mkdir -p ../results/grid/

$(EXE).o:$(EXE).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o $(EXE).out
