### CudaCanny Edge Detector

A high-performance implementation of the Canny edge detector with both CPU and CUDA GPU paths. It loads PGM images, runs Gaussian smoothing, gradient, non-maximum suppression, and hysteresis, then saves CPU and GPU edge maps while reporting timing and speedup.

#### Features
- CPU and CUDA implementations side-by-side
- Benchmarking with timing and speedup
- Simple Makefile build

#### Requirements
- CUDA Toolkit (nvcc)
- GCC/Clang for C sources

#### Build
```bash
make
```
Binaries are placed in `bin/`.

#### Run
```bash
./bin/canny canny/pics/pic_large.pgm 2.5 0.25 0.5
```
Adjust sigma, tlow, thigh as needed. Example images are in `canny/pics/`.

#### Project Layout
- `src/`: C/CUDA sources
- `include/`: headers
- `bin/`: compiled binary
- `build/`: object files
- `canny/pics/`: sample images (PGM)
