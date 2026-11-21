CC := gcc
CFLAGS := -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra

.PHONY: all clean

all: stream_bench

stream_bench: stream_bench.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f stream_bench
