CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS = -lm

TARGET = su2_sim
SRC = src/main.cpp

all: 
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)
clean:
	rm -f $(Target)
