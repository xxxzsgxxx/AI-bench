#!/bin/bash
#by scott, 20250628
#
#docker run --gpus all -it --pid=host --network=host nvcr.io/nvidia/pytorch:25.01-py3
#git clone --recursive https://github.com/NVIDIA/cutlass.git

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CMAKE_CUDA_ARCHITECTURES=120

CUTLASSDIR="/root/src/cutlass/"

##1 Peak FP32 TFLOPS (non-tensor)
cd $CUTLASSDIR
mkdir build-fp32 && cd build-fp32
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sgemm_256x128_8x5_*_align1
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./fp32.csv

##2 Peak FP16 TFLOPS (non-tensor)
cd $CUTLASSDIR
mkdir build-fp16 && cd build-fp16
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_hgemm_256x128_8x2_*_align1
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./fp16.csv

##3 Peak FP16 Tensor TFLOPS (FP16Acc)
cd $CUTLASSDIR
mkdir build-fp16_fp16acc && cd build-fp16_fp16acc
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_h16816gemm_256x128_32x3_*_align8
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./fp16_fp16acc.csv

##4 Peak FP16 Tensor TFLOPS (FP32Acc)
cd $CUTLASSDIR
mkdir build-fp16_fp32acc && cd build-fp16_fp32acc
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_256x128_32x3_*_align8
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./fp16_fp32acc.csv

##5 Peak BF16 Tensor TFLOPS (FP32Acc)
cd $CUTLASSDIR
mkdir build-bf16 && cd build-bf16
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_bf16_256x128_32x3_*_align8
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./bf16.csv

##6 Peak TF32 Tensor TFLOPS  cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_*_align4
cd $CUTLASSDIR
mkdir build-tf32 && cd build-tf32
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_*_align4
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./tf32.csv

##7 Peak INT8 Tensor TOPS          cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_*_aligin16
cd $CUTLASSDIR
mkdir build-int8 && cd build-int8
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_*_aligin16
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./int8.csv

##8 Peak INT4 Tensor TOPS   cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_*_align32
cd $CUTLASSDIR
mkdir build-int4 && cd build-int4
cmake .. -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_*_align32
make cutlass_profiler -j99

./tools/profiler/cutlass_profiler --operation=Gemm \
  --m=1024,2048,4096,8192,16384 \
  --n=1024,2048,4096,8192,16384 \
  --k=1024,2048,4096,8192,16384 \
  --output=./int4.csv
