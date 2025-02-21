all: clean
	clang main.c -o c 
clean:
	rm -rf c.bin

test: all
	./c.bin input01 && \
		./c.bin input02 && \
		./c.bin input03 && \
		./c.bin input04 && \
		./c.bin input05

