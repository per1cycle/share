all: clean
	clang main.c -o c 
clean:
	rm -rf c.bin

test: all
	./c input01 && \
		./c input02 && \
		./c input03 && \
		./c input04 && \
		./c input05

