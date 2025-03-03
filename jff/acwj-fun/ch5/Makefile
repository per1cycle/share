all: clean
	clang main.c -o c.bin
clean:
	rm -rf *.bin c.codegen

test: all
	./c.bin input01 && \
		./c.bin input02 && \
		./c.bin input03 && \
		./c.bin input04 && \
		./c.bin input05

