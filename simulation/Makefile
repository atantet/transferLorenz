INCDIR=-I$(HOME)/local/include/
LIBDIR=-L$(HOME)/local/lib/
LIBS=-lgsl -lgslcblas -lconfig++ -lergopack -lgomp
CONFIG=../cfg/readConfig

CC=icc
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -O3
LDFLAGSSTD=
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

all:$(EXE).out
	mkdir -p ../results/simulation

$(EXE).o:$(EXE).cpp
	$(CC) $(OMPFLAGS) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) $(OMPFLAGS) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o
	$(CC) $(OMPFLAGS) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o $(EXE).out
