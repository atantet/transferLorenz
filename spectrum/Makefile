INCDIR=-I$(HOME)/local/include/
LIBDIR=-L$(HOME)/local/lib/
<<<<<<< HEAD
LIBS=-lgsl -lconfig++ -lergopack -lgomp -larpack
#-lgslcblas
=======
LIBS=-lgsl -lconfig++ -lopenblas -larpack -lergopack
>>>>>>> fd01e4fd1720162ee0dde431bafdd1e8ec5d14c3
CONFIG=../cfg/readConfig

CC=icc -openmp
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -O3
LDFLAGSSTD=
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

EXE=spectrum

all:$(EXE).out
	mkdir -p ../results/spectrum/eigval
	mkdir -p ../results/spectrum/eigvec
	mkdir -p ../results/spectrum/conditionNumbers

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
