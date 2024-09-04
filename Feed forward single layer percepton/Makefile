CC := g++
CFLAGS := -Wall -Wextra -std=c++11
SRCS1 := main.cc
SRCS2 := regressionMatrix.cc
TARGET1 := neural_network
TARGET2 := neural_network_reg

.PHONY: all clean

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SRCS1)
	$(CC) $(CFLAGS) -o $(TARGET1) $(SRCS1)

$(TARGET2): $(SRCS2)
	$(CC) $(CFLAGS) -o $(TARGET2) $(SRCS2)

clean:
	rm -f $(TARGET1) $(TARGET2)
