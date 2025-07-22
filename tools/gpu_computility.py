# GPU computility Test.
# by scott, 20250610

import torch
import time
import logging
import platform
import subprocess
import sys
import os
import gc
import psutil
import pynvml
from datetime import datetime

# 设置显存分配策略减少碎片化
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('GPU_Precision_Benchmark.log')
    ]
)
logger = logging.getLogger()

# 精度类型与计算能力对应表
SUPPORT_MATRIX = {
    'fp64': 3.0,
    'tf32': 8.0,
    'fp32': 5.0,
    'fp16': 5.3,
    'bf16': 8.0,
    'int8': 7.5,
    'int4': 8.9,
    'fp8': 8.9,
    'bf8': 8.9,
    'fp4': 8.9,
    'bf4': 8.9
}

# 精度类型映射
DTYPE_MAPPING = {
    'fp64': torch.float64,
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'int8': torch.int8,
    'int4': torch.quint4x2,
    'tf32': torch.float32,
    'fp8': None,
    'bf8': None,
    'fp4': None,
    'bf4': None
}

# GPU核心规格数据库
GPU_SPECS = {
    "NVIDIA GeForce RTX 5090 D": {
        "cuda_cores": 16384,
        "tensor_cores": 512,
        "rt_cores": 128,
        "base_clock": 2235,
        "boost_clock": 2520
    },
    "NVIDIA GeForce RTX 4090": {
        "cuda_cores": 16384,
        "tensor_cores": 512,
        "rt_cores": 128,
        "base_clock": 2235,
        "boost_clock": 2520
    },
    "NVIDIA A100": {
        "cuda_cores": 6912,
        "tensor_cores": 432,
        "rt_cores": 0,
        "base_clock": 1095,
        "boost_clock": 1410
    },
    "NVIDIA RTX 6000 Ada": {
        "cuda_cores": 18176,
        "tensor_cores": 568,
        "rt_cores": 142,
        "base_clock": 915,
        "boost_clock": 2505
    }
}

# ================== 核心函数：计算能力到核心数映射 ==================
def compute_capability_to_cores(major, minor):
    """根据GPU架构计算CUDA核心数"""
    mp_count = torch.cuda.get_device_properties(0).multi_processor_count
    arch_code = 10 * major + minor
    
    # 架构核心映射表 
    cores_per_mp = {
        80: 128,   # Ampere (A100)
        86: 128,   # Ampere (GA102)
        87: 128,   # Ampere (AD102)
        89: 128,   # Blackwell (GB100)
        90: 128    # Hopper
    }
    return mp_count * cores_per_mp.get(arch_code, 64)

# ================== 自定义量化函数 ==================
def quantize_to_fp8(tensor: torch.Tensor, fmt: str = 'e4m3') -> torch.Tensor:
    """FP32张量量化为FP8格式"""
    if fmt not in ['e4m3', 'e5m2']:
        raise ValueError(f"Unsupported FP8 format: {fmt}")
    
    tensor_flat = tensor.view(-1)
    signs = torch.sign(tensor_flat)
    abs_tensor = torch.abs(tensor_flat)
    
    exponents = torch.floor(torch.log2(abs_tensor.clamp(min=1e-10)))
    mantissas = abs_tensor / (2 ** exponents)
    
    if fmt == 'e4m3':
        exp_bits, mant_bits = 4, 3
        exp_bias, max_exp = 7, 15
    else:  # e5m2
        exp_bits, mant_bits = 5, 2
        exp_bias, max_exp = 15, 31
    
    exponents = torch.clamp(exponents + exp_bias, 0, max_exp)
    mantissas = torch.clamp(torch.round(mantissas * (2 ** mant_bits)) / (2 ** mant_bits), 0.0, 1.0)
    
    quantized = signs * mantissas * (2 ** (exponents - exp_bias))
    return quantized.view_as(tensor)

def quantize_to_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """FP32张量量化为FP4格式（E2M1）"""
    tensor_flat = tensor.view(-1)
    signs = torch.sign(tensor_flat)
    abs_tensor = torch.abs(tensor_flat)
    
    exponents = torch.floor(torch.log2(abs_tensor.clamp(min=1e-10)))
    mantissas = abs_tensor / (2 ** exponents)
    
    exp_bias = 3
    max_exp = 3
    
    exponents = torch.clamp(exponents + exp_bias, 0, max_exp)
    mantissas = torch.clamp(torch.round(mantissas * 2) / 2, 0.0, 1.0)
    
    quantized = signs * mantissas * (2 ** (exponents - exp_bias))
    return quantized.view_as(tensor)

def quantize_to_bf8(tensor: torch.Tensor) -> torch.Tensor:
    """模拟BF8量化（类似BF16但为8位）"""
    bf16_tensor = tensor.to(torch.bfloat16)
    bf16_flat = bf16_tensor.view(-1)
    int_repr = bf16_flat.view(torch.int16)
    
    sign_mask = 0x8000
    exp_mask = 0x7C00
    mant_mask = 0x0300
    
    int_repr_bf8 = (int_repr & sign_mask) | ((int_repr & exp_mask) >> 3) | ((int_repr & mant_mask) >> 7)
    bf8_tensor = int_repr_bf8.view(torch.int16).view(torch.bfloat16)
    return bf8_tensor.to(tensor.dtype)

def quantize_to_bf4(tensor: torch.Tensor) -> torch.Tensor:
    """模拟BF4量化（非标准格式）"""
    bf8_tensor = quantize_to_bf8(tensor)
    return quantize_to_fp4(bf8_tensor)

# ================== 系统信息收集 ==================
def get_supported_precisions(compute_cap):
    """修复：确保总是返回列表类型"""
    try:
        if compute_cap is None:
            return []  # 处理None值情况
        return [p for p, cc in SUPPORT_MATRIX.items() if compute_cap >= cc]
    except Exception as e:
        logger.error(f"精度支持检测失败: {str(e)}")
        return []  # 确保返回可迭代对象

def get_system_info():
    """收集详细的系统环境信息"""
    info = {}
    
    # 操作系统信息
    info['os'] = platform.system()
    info['os_release'] = platform.release()
    info['os_version'] = platform.version()
    info['platform'] = platform.platform()
    
    # Python环境
    info['python'] = sys.version
    info['python_compiler'] = platform.python_compiler()
    
    # PyTorch环境
    info['torch'] = torch.__version__
    info['cuda'] = torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'
    info['cudnn'] = torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'
    
    # 硬件信息
    info['architecture'] = platform.architecture()[0]
    info['processor'] = platform.processor()
    info['machine'] = platform.machine()
    
    # 内存信息
    mem = psutil.virtual_memory()
    info['sys_mem'] = f"{mem.total//1024**3} GB"
    
    # CPU信息
    info['cpu_cores'] = os.cpu_count()
    info['cpu_percent'] = f"{psutil.cpu_percent()}%"
    
    # 获取NVIDIA驱动信息
    try:
        nvidia_smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        if nvidia_smi:
            driver_info = nvidia_smi.split(',')
            info['driver'] = driver_info[0].strip()
            info['gpu_name'] = driver_info[1].strip()
            info['gpu_mem'] = driver_info[2].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info['driver'] = 'N/A'
        info['gpu_name'] = 'N/A'
        info['gpu_mem'] = 'N/A'
    
    # 获取CUDA编译器信息
    try:
        nvcc_version = subprocess.check_output(
            ['nvcc', '--version'], 
            stderr=subprocess.STDOUT
        ).decode('utf-8')
        info['nvcc'] = nvcc_version.split('\n')[-2].split(',')[-1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info['nvcc'] = 'N/A'
    
    # 获取glibc版本
    try:
        ldd_version = subprocess.check_output(
            ['ldd', '--version'], 
            stderr=subprocess.STDOUT
        ).decode('utf-8')
        info['glibc'] = ldd_version.split('\n')[0].split()[-1]
    except (subprocess.CalledProcessError, FileNotFoundError):
        info['glibc'] = 'N/A'
    
    # 收集关键环境变量
    info['env_vars'] = dict(os.environ)
    
    # ============= 增强GPU信息获取 =============
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # 1. 获取实时时钟频率 
        clocks = {
            'graphics': pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
            'sm': pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            'mem': pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
            'video': pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_VIDEO)
        }
        max_clocks = {
            'graphics': pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
            'sm': pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        }
        
        # 2. 动态计算CUDA核心数 
        major, minor = torch.cuda.get_device_capability(0)
        cuda_cores = compute_capability_to_cores(major, minor)
        
        # 3. 获取Tensor核心数（静态数据库）
        gpu_name = info['gpu_name'].strip()
        tensor_cores = GPU_SPECS.get(gpu_name, {}).get("tensor_cores", "N/A")
        
        # 4. 获取其他关键信息 
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # 5. 保存到info字典
        info['gpu_cores'] = {
            'cuda_cores': cuda_cores,
            'tensor_cores': tensor_cores,
            'rt_cores': GPU_SPECS.get(gpu_name, {}).get("rt_cores", "N/A")
        }
        info['gpu_clocks'] = {
            'current': clocks,
            'max': max_clocks,
            'temperature': temp,
            'power_usage': f"{power_usage:.1f} W",
            'power_limit': f"{power_limit:.1f} W",
            'gpu_utilization': f"{utilization.gpu}%",
            'mem_utilization': f"{utilization.memory}%"
        }
        
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.error(f"GPU信息获取失败: {str(e)}")
        info['gpu_cores'] = {'cuda_cores': 'N/A', 'tensor_cores': 'N/A', 'rt_cores': 'N/A'}
        info['gpu_clocks'] = 'N/A'
    
    return info

# ================== 辅助函数 ==================
def log_tensor_info(tensor, name):
    """记录张量信息"""
    if tensor is not None:
        dtype = str(tensor.dtype).split('.')[-1]
        device = tensor.device
        shape = tensor.shape
        logger.info(f"{name}张量: dtype={dtype}, shape={shape}, device={device}")
    else:
        logger.info(f"{name}张量: None")

# ================== 基准测试函数 ==================
def benchmark(precision, device):
    """执行指定精度的基准测试"""
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"===== 开始 {precision.upper()} 精度测试 =====")
    
    # 检查精度支持
    if precision in DTYPE_MAPPING and DTYPE_MAPPING[precision] is None:
        err_msg = f"当前PyTorch版本({torch.__version__})不支持{precision.upper()}精度"
        logger.error(err_msg)
        return 0, err_msg

    # 获取显存状态
    total_mem = torch.cuda.get_device_properties(device).total_memory
    allocated_mem = torch.cuda.memory_allocated(device)
    free_mem = total_mem - allocated_mem
    logger.info(f"总显存: {total_mem/1024**3:.2f} GB, 已用显存: {allocated_mem/1024**3:.2f} GB, 可用显存: {free_mem/1024**3:.2f} GB")
    
    # 根据精度调整矩阵大小（动态安全系数）
    safe_factor = 0.6 if precision == 'fp64' else 0.7
    element_size = 4  # 所有自定义量化最终转换为FP32计算
    
    # 计算最大可能的矩阵尺寸
    max_elements = (free_mem * safe_factor) / (element_size * 3)
    matrix_size = int(max_elements ** 0.5)
    matrix_size = (matrix_size // 256) * 256
    
    # 根据精度设置上限
    upper_limit = 16384 if precision in ['int8', 'int4', 'fp8', 'bf8', 'fp4', 'bf4'] else (24576 if precision != 'fp64' else 16384)
    matrix_size = min(matrix_size, upper_limit)
    
    logger.info(f"测试矩阵规模: {matrix_size}x{matrix_size}")

    try:
        # 特殊精度处理
        if precision == 'int8':
            logger.info("创建INT8张量...")
            a = torch.randint(-128, 127, (matrix_size, matrix_size), dtype=torch.int8, device=device)
            b = torch.randint(-128, 127, (matrix_size, matrix_size), dtype=torch.int8, device=device)
            log_tensor_info(a, "A")
            log_tensor_info(b, "B")
            
            logger.info("INT8预热计算...")
            _ = torch.matmul(a.float(), b.float())
            torch.cuda.empty_cache()
            
            # 基准测试
            logger.info("开始INT8基准测试...")
            start = time.time()
            for i in range(100):
                c = torch.matmul(a.float(), b.float())
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

        elif precision == 'int4':
            try:
                logger.info("创建INT4张量...")
                scale = 0.05
                zero_point = 0
                
                a_data = torch.randn(matrix_size, matrix_size, device=device)
                b_data = torch.randn(matrix_size, matrix_size, device=device)
                
                a = torch.quantize_per_tensor(
                    a_data,
                    scale=scale,
                    zero_point=zero_point,
                    dtype=torch.quint4x2
                )
                b = torch.quantize_per_tensor(
                    b_data,
                    scale=scale,
                    zero_point=zero_point,
                    dtype=torch.quint4x2
                )
                log_tensor_info(a, "A")
                log_tensor_info(b, "B")
                
                logger.info("INT4预热计算...")
                a_deq = a.dequantize().to(torch.float32)
                b_deq = b.dequantize().to(torch.float32)
                _ = torch.matmul(a_deq, b_deq)
                del a_deq, b_deq
                torch.cuda.empty_cache()
                
                logger.info("开始INT4基准测试...")
                start = time.time()
                for i in range(100):
                    a_deq = a.dequantize().to(torch.float32)
                    b_deq = b.dequantize().to(torch.float32)
                    c = torch.matmul(a_deq, b_deq)
                    if (i + 1) % 10 == 0:
                        logger.info(f"已完成 {i+1}/100 次计算")
                    del a_deq, b_deq, c
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                log_tensor_info(c, "结果")
                
            except Exception as e:
                err_msg = f"int4测试失败: {str(e)}"
                logger.error(err_msg)
                if "CUDA" in str(e):
                    logger.error("可能原因：当前PyTorch/CUDA版本不支持INT4直接计算，尝试更新驱动或使用量化专用库")
                return 0, err_msg

        elif precision == 'tf32':
            if torch.cuda.is_available():
                logger.info("启用TF32支持...")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning("当前设备不支持TF32")

            dtype = DTYPE_MAPPING[precision]
            logger.info("创建TF32张量...")
            a = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            log_tensor_info(a, "A")
            log_tensor_info(b, "B")
            
            logger.info("TF32预热计算...")
            _ = torch.mm(a, b)
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("开始TF32基准测试...")
            start = time.time()
            for i in range(100):
                c = torch.mm(a, b)
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

        # FP8精度测试
        elif precision == 'fp8':
            logger.info("创建FP8张量...")
            a_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            b_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            
            logger.info("执行FP8量化...")
            a_fp8 = quantize_to_fp8(a_fp32, fmt='e4m3')
            b_fp8 = quantize_to_fp8(b_fp32, fmt='e4m3')
            log_tensor_info(a_fp8, "A_FP8")
            log_tensor_info(b_fp8, "B_FP8")
            
            logger.info("FP8预热计算...")
            _ = torch.matmul(a_fp8.to(torch.float32), b_fp8.to(torch.float32))
            torch.cuda.empty_cache()
            
            logger.info("开始FP8基准测试...")
            start = time.time()
            for i in range(100):
                a_fp32_deq = a_fp8.to(torch.float32)
                b_fp32_deq = b_fp8.to(torch.float32)
                c = torch.matmul(a_fp32_deq, b_fp32_deq)
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

        # FP4精度测试
        elif precision == 'fp4':
            logger.info("创建FP4张量...")
            a_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            b_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            
            logger.info("执行FP4量化...")
            a_fp4 = quantize_to_fp4(a_fp32)
            b_fp4 = quantize_to_fp4(b_fp32)
            log_tensor_info(a_fp4, "A_FP4")
            log_tensor_info(b_fp4, "B_FP4")
            
            logger.info("FP4预热计算...")
            _ = torch.matmul(a_fp4.to(torch.float32), b_fp4.to(torch.float32))
            torch.cuda.empty_cache()
            
            logger.info("开始FP4基准测试...")
            start = time.time()
            for i in range(100):
                a_fp32_deq = a_fp4.to(torch.float32)
                b_fp32_deq = b_fp4.to(torch.float32)
                c = torch.matmul(a_fp32_deq, b_fp32_deq)
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {极+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

        # BF8精度测试
        elif precision == 'bf8':
            logger.info("创建BF8张量...")
            a_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            b_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            
            logger.info("执行BF8量化...")
            a_bf8 = quantize_to_bf8(a_fp32)
            b_bf8 = quantize_to_bf8(b极f32)
            log_tensor_info(a_bf8, "A_BF8")
            log_tensor_info(b_bf8, "B_BF8")
            
            logger.info("BF8预热计算...")
            _ = torch.matmul(a_bf8.to(torch.float32), b_bf8.to(torch.float32))
            torch.cuda.empty_cache()
            
            logger.info("开始BF8基准测试...")
            start = time.time()
            for i in range(100):
                a_fp32_deq = a_bf8.to(torch.float32)
                b_fp32_deq = b_bf8.to(torch.float32)
                c = torch.matmul(a_fp32_deq, b_fp32_deq)
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

        # BF4精度测试
        elif precision == 'bf4':
            logger.info("创建BF4张量...")
            a_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            b_fp32 = torch.randn(matrix_size, matrix_size, device=device)
            
            logger.info("执行BF4量化...")
            a_bf4 = quantize_to_bf4(a_fp32)
            b_bf4 = quantize_to_bf4(b_fp32)
            log_tensor_info(a_bf4, "A_BF4")
            log_tensor_info(b_bf4, "B_BF4")
            
            logger.info("BF4预热计算...")
            _ = torch.matmul(a_bf4.to(torch.float32), b_bf4.to(torch.float32))
            torch.cuda.empty_cache()
            
            logger.info("开始BF4基准测试...")
            start = time.time()
            for i in range(100):
                a_fp32_deq = a_bf4.to(torch.float32)
                b_fp32_deq = b_bf4.to(torch.float32)
                c = torch.matmul(a_fp32_deq, b_fp32_deq)
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

        # 标准浮点精度处理
        else:
            dtype = DTYPE_MAPPING[precision]
            logger.info(f"创建{precision.upper()}张量...")
            a = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            log_tensor_info(a, "A")
            log_tensor_info(b, "B")
            
            logger.info(f"{precision.upper()}预热计算...")
            _ = torch.mm(a, b)
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"开始{precision.upper()}基准测试...")
            start = time.time()
            for i in range(100):
                c = torch.mm(a, b)
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/100 次计算")
            torch.cuda.synchronize()
            elapsed = time.time() - start
            log_tensor_info(c, "结果")

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning(f"{precision.upper()}精度测试遇到显存不足，尝试减小矩阵规模...")
            new_size = int(matrix_size * 0.8)
            new_size = (new_size // 256) * 256
            if new_size < 256:
                err_msg = f"矩阵规模过小({new_size}x{new_size})无法测试"
                logger.error(err_msg)
                return 0, err_msg
            logger.info(f"新矩阵规模: {new_size}x{new_size}")
            return benchmark(precision, device)
        else:
            err_msg = f"测试失败: {str(e)}"
            logger.error(err_msg)
            return 0, err_msg
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    # 计算FLOPS
    flops = 2 * matrix_size ** 3 * 100 / elapsed if elapsed > 0 else 0
    tflops = flops / 1e12
    logger.info(f"{precision.upper()}测试完成: {tflops:.2f} TFLOPS (耗时: {elapsed:.2f}秒)")
    return tflops, "成功"

# ================== 结果输出函数 ==================
def log_performance_summary(results):
    """输出性能数据汇总"""
    logger.info("\n===== 性能数据汇总 =====")
    for prec, res in results.items():
        if res['status'] == "成功":
            logger.info(f"{prec.upper():<6}: {res['tflops']:.2f} TFLOPS")
    
    # 计算相对性能比
    if 'fp32' in results and results['fp32']['status'] == "成功":
        fp32_tflops = results['fp32']['tflops']
        for prec in ['fp64', 'fp16', 'bf16', 'int8', 'int4']:
            if prec in results and results[prec]['status'] == "成功":
                ratio = results[prec]['tflops'] / fp32_tflops
                logger.info(f"{prec.upper()}/FP32性能比: {ratio:.2f}x")
    
    # 特殊精度比较
    if 'fp16' in results and 'bf16' in results and \
        results['fp16']['status'] == "成功" and results['bf16']['status'] == "成功":
        ratio = results['bf16']['tflops'] / results['fp16']['tflops']
        logger.info(f"BF16/FP16性能比: {ratio:.2f}x")

def log_environment_summary(sys_info):
    """输出环境变量摘要"""
    logger.info("\n===== 环境变量摘要 =====")
    env_vars = sys_info.get('env_vars', {})
    
    # 分类整理环境变量
    python_vars = {k: v for k, v in env_vars.items() if 'PYTHON' in k}
    cuda_vars = {k: v for k, v in env_vars.items() if 'CUDA' in k}
    path_vars = {k: v for k, v in env_vars.items() if 'PATH' in k}
    other_vars = {k: v for k, v in env_vars.items() 
                  if 'PYTHON' not in k and 'CUDA' not in k and 'PATH' not in k}
    
    logger.info("\n--- Python相关环境变量 ---")
    for var, value in python_vars.items():
        logger.info(f"{var}: {value}")
    
    logger.info("\n--- CUDA相关环境变量 ---")
    for var, value in cuda_vars.items():
        logger.info(f"{var}: {value}")
    
    logger.info("\n--- 路径相关环境变量 ---")
    for var, value in path_vars.items():
        logger.info(f"{var}: {value}")
    
    logger.info("\n--- 其他重要环境变量 ---")
    for var, value in other_vars.items():
        logger.info(f"{var}: {value}")

def highlight_performance_results(results):
    """高亮显示性能结果"""
    logger.info("\n\033[1;36m===== 精度性能报告 =====\033[0m")
    for prec, res in results.items():
        status = res['status']
        if status == "成功":
            logger.info(f"\033[1;32m{prec.upper():<6}: {res['tflops']:.2f} TFLOPS\033[0m")
        else:
            logger.info(f"\033[1;31m{prec.upper():<6}: {status}\033[0m")

def log_detailed_system_info(sys_info):
    """输出详细的系统环境信息"""
    logger.info("\n\033[1;35m===== 详细系统环境信息 =====\033[0m")
    logger.info(f"操作系统: {sys_info['os']} {sys_info['os_release']} ({sys_info['os_version']})")
    logger.info(f"平台架构: {sys_info['architecture']}")
    logger.info(f"处理器: {sys_info['processor']}")
    logger.info(f"CPU核心数: {sys_info['cpu_cores']}")
    logger.info(f"系统内存: {sys_info['sys_mem']}")
    logger.info(f"Python版本: {sys_info['python'].split()[0]}")
    logger.info(f"PyTorch版本: {sys_info['torch']}")
    logger.info(f"CUDA版本: {sys_info['cuda']}")
    logger.info(f"cuDNN版本: {sys_info['cudnn']}")
    logger.info(f"glibc版本: {sys_info['glibc']}")
    
    # GPU详细信息
    logger.info("\n\033[1;35m===== GPU硬件规格 =====\033[0m")
    logger.info(f"显卡型号: {sys_info.get('gpu_name', 'N/A')}")
    logger.info(f"显存容量: {sys_info.get('gpu_mem', 'N/A')}")
    
    # 核心改进：输出计算核心详情
    cores_info = sys_info.get('gpu_cores', {})
    logger.info(f"CUDA核心数: {cores_info.get('cuda_cores', 'N/A')}")
    logger.info(f"Tensor核心数: {cores_info.get('tensor_cores', 'N/A')}")
    logger.info(f"RT核心数: {cores_info.get('rt_cores', 'N/A')}")
    
    # 核心改进：输出实时时钟信息
    if 'gpu_clocks' in sys_info and isinstance(sys_info['gpu_clocks'], dict):
        clocks = sys_info['gpu_clocks']
        logger.info("\n\033[1;35m===== 实时GPU状态 =====\033[0m")
        logger.info(f"当前图形时钟: {clocks['current']['graphics']} MHz")
        logger.info(f"当前SM时钟: {clocks['current']['sm']} MHz")
        logger.info(f"当前显存时钟: {clocks['current']['mem']} MHz")
        logger.info(f"最大SM时钟: {clocks['max']['sm']} MHz")
        logger.info(f"当前温度: {clocks['temperature']}°C")
        logger.info(f"当前功耗: {clocks['power_usage']}")
        logger.info(f"功耗限制: {clocks['power_limit']}")
        logger.info(f"GPU利用率: {clocks['gpu_utilization']}")
        logger.info(f"显存利用率: {clocks['mem_utilization']}")

def log_supported_precisions(compute_cap):
    """输出支持精度列表"""
    logger.info("\n\033[1;35m===== 支持精度列表 =====\033[0m")
    for p, min_cc in SUPPORT_MATRIX.items():
        status = "✅" if compute_cap >= min_cc else "❌"
        logger.info(f"{p.upper():<6}: 最小计算能力 {min_cc} {status}")

def log_test_methodology():
    """输出测试方法论说明"""
    logger.info("\n\033[1;33m===== 测试方法论说明 =====\033[0m")
    logger.info("1. 测试核心：使用大型矩阵乘法(GEMM)操作评估GPU计算能力")
    logger.info("2. 精度处理：")
    logger.info("   - FP32/FP16/BF16: 原生精度支持")
    logger.info("   - INT8/INT4: 使用PyTorch量化API")
    logger.info("   - FP8/FP4/BF8/BF4: 自定义量化函数实现")
    logger.info("3. 性能计算：")
    logger.info("   FLOPS = 2 * N³ * 100 / 时间(s)")
    logger.info("   TFLOPS = FLOPS / 10¹²")
    logger.info("4. 注意事项：")
    logger.info("   - 测试结果可能高于实际应用性能（纯计算无业务逻辑）")
    logger.info("   - gpu-burn等工具使用混合运算，结果通常低于本测试")

# ================== 主程序 ==================
if __name__ == '__main__':
    total_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        logger.error("CUDA设备未找到")
        exit()

    # 收集系统信息
    sys_info = get_system_info()
    
    # 记录系统信息
    logger.info("===== 系统环境信息 =====")
    logger.info(f"操作系统: {sys_info['os']} {sys_info['os_release']} ({sys_info['os_version']})")
    logger.info(f"平台: {sys_info['platform']}")
    logger.info(f"架构: {sys_info['architecture']}")
    logger.info(f"处理器: {sys_info['processor']}")
    logger.info(f"CPU核心: {sys_info['cpu_cores']}, 使用率: {sys_info['cpu_percent']}")
    logger.info(f"系统内存: {sys_info['sys_mem']}")
    logger.info(f"Python版本: {sys_info['python']}")
    logger.info(f"Python编译器: {sys_info['python_compiler']}")
    logger.info(f"glibc版本: {sys_info['glibc']}")
    logger.info(f"PyTorch版本: {sys_info['torch']}")
    logger.info(f"CUDA版本: {sys_info['cuda']}")
    logger.info(f"cuDNN版本: {sys_info['cudnn']}")
    logger.info(f"NVCC版本: {sys_info['nvcc']}")
    logger.info(f"显卡驱动: {sys_info['driver']}")
    logger.info(f"GPU型号: {sys_info['gpu_name']}")
    logger.info(f"GPU显存: {sys_info['gpu_mem']}")

    # 获取GPU信息
    try:
        compute_cap = float('.'.join(map(str, torch.cuda.get_device_capability(0))))
    except Exception as e:
        logger.error(f"获取GPU计算能力失败: {str(e)}")
        compute_cap = 0.0  # 设置安全默认值

    device_props = torch.cuda.get_device_properties(0)
    precisions = get_supported_precisions(compute_cap)
    
    # 确保precisions是列表类型
    if not isinstance(precisions, list):
        logger.warning("精度列表初始化失败，重置为空列表")
        precisions = []
    
    # 安全添加自定义精度到支持列表
    custom_precisions = ['fp8', 'bf8', 'fp4', 'bf4']
    for cp in custom_precisions:
        # 确保SUPPORT_MATRIX中存在该精度
        if cp in SUPPORT_MATRIX:
            # 检查计算能力是否支持该精度
            if compute_cap >= SUPPORT_MATRIX[cp]:
                # 如果该精度不在precisions中，则添加
                if cp not in precisions:
                    precisions.append(cp)
            else:
                logger.info(f"计算能力{compute_cap}不支持{cp}精度，要求{SUPPORT_MATRIX[cp]}")
        else:
            logger.warning(f"未知精度类型 {cp}，跳过")
    
    logger.info("\n===== GPU 算力基准测试开始 =====")
    logger.info(f"计算能力: {compute_cap}")
    logger.info(f"显存容量: {device_props.total_memory/1024**3:.1f} GB")
    logger.info(f"支持的精度类型: {', '.join(precisions)}")

    # 设置TF32支持
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 执行基准测试
    results = {}
    for precision in precisions:
        test_start = time.time()
        try:
            logger.info(f"\n>>>>>> 启动 {precision.upper()} 测试 <<<<<<")
            tflops, status = benchmark(precision, device)
            results[precision] = {
                'tflops': tflops,
                'status': status,
                'duration': time.time() - test_start
            }
            if status == "成功":
                logger.info(f"{precision.upper()}测试成功: {tflops:.2f} TFLOPS (耗时: {results[precision]['duration']:.2f}秒)")
            else:
                logger.warning(f"{precision.upper()}测试失败: {status}")
        except Exception as e:
            err_msg = f"测试失败: {str(e)}"
            logger.error(err_msg)
            results[precision] = {
                'tflops': 0,
                'status': err_msg,
                'duration': time.time() - test_start
            }
        finally:
            # 每次测试后彻底清理显存
            torch.cuda.empty_cache()
            gc.collect()

    # 输出结果
    logger.info("\n===== 基准测试结果 =====")
    for prec, res in results.items():
        status = res['status']
        if status == "成功":
            logger.info(f"{prec.upper():<10}: {res['tflops']:.2f} TFLOPS (耗时: {res['duration']:.2f}秒)")
        else:
            logger.warning(f"{prec.upper():<10}: {status}")
 
    # 输出测试方法论说明
    log_test_methodology()

    # 输出环境变量摘要
    log_environment_summary(sys_info)

    # 最终报告
    architecture = 'Blackwell' if 'Blackwell' in device_props.name else f'Compute {compute_cap}'
    logger.info("\n===== GPU 算力基准测试报告 =====")
    logger.info(f"设备架构: {architecture}")
    logger.info(f"设备型号: {device_props.name}")
    logger.info(f"显存容量: {device_props.total_memory/1024**3:.1f} GB")
    logger.info(f"总运行时间: {time.time()-total_start:.2f}秒")
    logger.info(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 输出详细系统信息和GPU规格
    log_detailed_system_info(sys_info)
    
    # 输出支持精度列表
    log_supported_precisions(compute_cap)
 
     # 高亮显示性能结果
    highlight_performance_results(results)   
    
    # 性能数据汇总
    log_performance_summary(results)
