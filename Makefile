CXX=g++
CXXFLAGS=-std=c++11 -Wall -Werror -Iinclude
DATASET=mnist/train-images-idx3-ubyte mnist/train-labels-idx1-ubyte mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte

knn: $(DATASET) src/main.cc
	$(CXX) $(CXXFLAGS) -o $@ src/main.cc

debug: $(DATASET) src/main.cc
	$(CXX) -DDEBUG $(CXXFLAGS) -o $@ src/main.cc

$(DATASET):
	mkdir -p mnist && curl http://yann.lecun.com/exdb/$@.gz | gunzip -c > $@

clean:
	rm -rf knn debug $(DATASET) mnist