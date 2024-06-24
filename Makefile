main:
	g++ -I.\include -L.\lib -o src/main.exe src/main.cpp -lSDL2main -lSDL2 -lglew32 -lopengl32
	src/main.exe

test:
	nvcc -lcublas src/test.cu -o src/test.exe
	src/test.exe

clean:
	rm -f */*.exe