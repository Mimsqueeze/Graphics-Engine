all: main

main: test
	g++ -o src/main.exe src/main.cpp src/test.obj -I./include -L./lib -lSDL2main -lSDL2 -lglew32 -lopengl32

test:
	nvcc -c -o src/test.obj src/test.cpp -lcublas
	nvcc -shared -o src/test.dll src/test.obj -lcublas

clean:
	rm -f src/*.exe src/*.o src/*.lib src/*.exp src/*.obj src/test.dll
