all:
	g++ -I.\include -L.\lib -o src/main.exe src/main.cpp -lSDL2main -lSDL2 -lglew32 -lopengl32
	src/main.exe

clean:
	rm -f */*.exe