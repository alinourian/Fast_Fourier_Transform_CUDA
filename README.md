# Fast_Fourier_Transform_CUDA
Parallel FFT algorithm using CUDA (Cooley-Tukey algorithm)


Compile: 
```console
nvcc -O2 fft_main.cu fft.cu -o fft
```
Execute: 
```console
./fft M
```

Note that $N=2^M$ and $M = 23$, $24$, $25$ and $26$.
