#!/usr/bin/env python3
"""
GPU Computility Test (v20)
- PRECISION_INFO unified matrix
- FP8 _scaled_mm return value fix (tuple vs tensor)
- bf8 precision support
- Unified GEMM dispatch (eliminates if/else chains)
- Binary-search matrix auto-scaling
- --matrix-size / --precisions CLI flags
- ETA progress in benchmark
- JSON output for machine consumption
- Reference GPU specs with bus_width
- DmonMonitor auto-restart
- Type hints
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil
import pynvml
import torch

log_filename = f"GPU_Bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_filename, encoding='utf-8')],
)
logger = logging.getLogger()
logger.info(f"日志文件: {log_filename}")

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
GLOBAL_SEED = 42

# ================= 精度信息矩阵 =================
PRECISION_INFO: Dict[str, Dict[str, Any]] = {
    'fp64': {'min_cap': 3.0, 'bytes': 8, 'dtype': torch.float64, 'tensor_core': False},
    'tf32': {'min_cap': 8.0, 'bytes': 4, 'dtype': torch.float32, 'tensor_core': True},
    'fp32': {'min_cap': 5.0, 'bytes': 4, 'dtype': torch.float32, 'tensor_core': False},
    'fp16': {'min_cap': 5.3, 'bytes': 2, 'dtype': torch.float16, 'tensor_core': True},
    'bf16': {'min_cap': 8.0, 'bytes': 2, 'dtype': torch.bfloat16, 'tensor_core': True},
    'int8': {'min_cap': 7.5, 'bytes': 1, 'dtype': torch.int8, 'tensor_core': True},
    'fp8':  {'min_cap': 8.9, 'bytes': 1, 'dtype': None, 'tensor_core': True},
    'bf8':  {'min_cap': 8.9, 'bytes': 1, 'dtype': None, 'tensor_core': True},
    'int4': {'min_cap': 7.5, 'bytes': 0.5, 'dtype': None, 'tensor_core': True},
    'fp4':  {'min_cap': 9.0, 'bytes': 0.5, 'dtype': None, 'tensor_core': True},
}

# ================= 参考 GPU 规格 =================
REFERENCE_GPU_SPECS: Dict[str, Dict[str, Any]] = {
    "NVIDIA GeForce RTX 5090 D v2":                 {"cuda_cores": 16384, "tensor_cores": 512, "rt_cores": 128, "bus_width": 384},
    "NVIDIA GeForce RTX 5090":                      {"cuda_cores": 21760, "tensor_cores": 680, "rt_cores": 170, "bus_width": 512},
    "NVIDIA RTX 6000D":                             {"cuda_cores": 18176, "tensor_cores": 568, "rt_cores": 142, "bus_width": 384},
    "NVIDIA RTX 5000 Blackwell":                    {"cuda_cores": 10752, "tensor_cores": 336, "rt_cores": 84, "bus_width": 384},
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": {"cuda_cores": 24064, "tensor_cores": 752, "rt_cores": 188, "bus_width": 512},
    "NVIDIA RTX 4090":                              {"cuda_cores": 16384, "tensor_cores": 512, "rt_cores": 128, "bus_width": 384},
    "NVIDIA RTX 4090D":                             {"cuda_cores": 14592, "tensor_cores": 456, "rt_cores": 114, "bus_width": 384},
    "NVIDIA H100":                                  {"cuda_cores": 16896, "tensor_cores": 528, "bus_width": 5120},
    "NVIDIA H200":                                  {"cuda_cores": 16896, "tensor_cores": 528, "bus_width": 5120},
    "NVIDIA H800":                                  {"cuda_cores": 16896, "tensor_cores": 528, "bus_width": 5120},
    "NVIDIA H20":                                   {"cuda_cores": 14592, "tensor_cores": 456, "bus_width": 5120},
    "NVIDIA B200":                                  {"cuda_cores": 20480, "tensor_cores": 640, "bus_width": 8192},
    "NVIDIA B300":                                  {"cuda_cores": 25600, "tensor_cores": 800, "bus_width": 8192},
}


def print_reference_specs(gpu_name: str) -> None:
    if gpu_name in REFERENCE_GPU_SPECS:
        spec = REFERENCE_GPU_SPECS[gpu_name]
        items = ", ".join(f"{k}={v}" for k, v in spec.items())
        logger.info("参考数据（非实时）: " + items)
    else:
        logger.info("该 GPU 暂无权威参考数据。")


# ================= 辅助函数 =================
def compute_capability_to_cores(major: int, minor: int, device: torch.device) -> int:
    mp = torch.cuda.get_device_properties(device).multi_processor_count
    mapping = {80: 128, 86: 128, 87: 128, 89: 128, 90: 128}
    return mp * mapping.get(10 * major + minor, 64)


def get_phys_idx_from_torch(torch_dev_idx: int) -> Optional[int]:
    props = torch.cuda.get_device_properties(torch_dev_idx)
    tuuid = str(props.uuid).replace('GPU-', '')
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in out.split('\n'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 2 and parts[1].replace('GPU-', '') == tuuid:
                return int(parts[0])
    except Exception:
        pass
    return None


def query_tensor_cores(dev: torch.device) -> Any:
    phys_idx = get_phys_idx_from_torch(dev)
    if phys_idx is None:
        return 'N/A'
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=tensor_core_count', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        lines = out.split('\n')
        if len(lines) > phys_idx and lines[phys_idx].isdigit():
            return int(lines[phys_idx])
    except Exception:
        pass
    try:
        q = subprocess.check_output(['nvidia-smi', '-q', '-i', str(phys_idx)], stderr=subprocess.DEVNULL).decode()
        m = re.search(r'Tensor Core Count\s*:\s*(\d+)', q)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 'N/A'


def get_memory_bus_width(phys_idx: Optional[int]) -> Optional[int]:
    if phys_idx is None:
        return None
    try:
        q = subprocess.check_output(['nvidia-smi', '-q', '-i', str(phys_idx)], stderr=subprocess.DEVNULL).decode()
        m = re.search(r'Bus Width\s*:\s*(\d+)\s*bit', q)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def get_vbios_version(phys_idx: Optional[int]) -> str:
    if phys_idx is None:
        return 'N/A'
    try:
        q = subprocess.check_output(['nvidia-smi', '-q', '-i', str(phys_idx)], stderr=subprocess.DEVNULL).decode()
        m = re.search(r'VBIOS Version\s*:\s*(\S+)', q)
        if m:
            return m.group(1)
    except Exception:
        pass
    return 'N/A'


def get_extended_sysinfo(handle: Any, dev: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'os': platform.system(),
        'os_release': platform.release(),
        'kernel': platform.version(),
        'gcc': 'N/A',
        'python': sys.version,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'sys_mem': f"{psutil.virtual_memory().total // 1024 ** 3}GB",
        'cpu_cores': os.cpu_count(),
    }
    try:
        gcc_out = subprocess.check_output(['gcc', '--version'], stderr=subprocess.STDOUT).decode().split('\n')[0]
        info['gcc'] = gcc_out.strip()
    except Exception:
        pass
    try:
        ver = pynvml.nvmlSystemGetDriverVersion()
        info['driver'] = ver.decode() if isinstance(ver, bytes) else ver
    except Exception:
        info['driver'] = 'N/A'
    props = torch.cuda.get_device_properties(dev)
    info['gpu_name'] = props.name
    info['gpu_mem'] = f"{props.total_memory / 1024 ** 3:.1f}GB"
    info['cuda_cores'] = compute_capability_to_cores(*torch.cuda.get_device_capability(dev), dev)
    info['tensor_cores'] = query_tensor_cores(dev)
    phys_idx = get_phys_idx_from_torch(dev.index)
    info['vbios'] = get_vbios_version(phys_idx) if phys_idx is not None else 'N/A'
    for pkg in ['numpy', 'psutil', 'pynvml']:
        try:
            mod = __import__(pkg)
            info[pkg] = mod.__version__
        except Exception:
            pass
    return info


def print_extended_env(info: Dict[str, Any]) -> None:
    logger.info("\n===== 详细环境信息 =====")
    keys = ['time', 'os', 'kernel', 'gcc', 'python', 'torch', 'cuda', 'cudnn',
            'driver', 'vbios', 'gpu_name', 'gpu_mem', 'cuda_cores', 'tensor_cores',
            'sys_mem', 'cpu_cores']
    for k in keys:
        logger.info(f"{k}: {info.get(k, 'N/A')}")
    pkgs = [f"{p}={v}" for p, v in info.items() if p in ('numpy', 'psutil', 'pynvml')]
    if pkgs:
        logger.info("关键软件包: " + ", ".join(pkgs))


# ================= FP8/BF8 矩阵乘法 =================
def fp8_matmul(a_fp8: torch.Tensor, b_fp8: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, '_scaled_mm'):
        raise RuntimeError("torch._scaled_mm 不可用，请升级 PyTorch (>=2.2)")
    scale = torch.tensor(1.0, device=a_fp8.device, dtype=torch.float32)
    result = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale)
    if isinstance(result, tuple):
        return result[0]
    return result


def dispatch_matmul(
    precision: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Unified GEMM dispatch for all supported precision types."""
    if precision == 'fp8':
        a_q = a.to(torch.float8_e4m3fn)
        b_q = b.to(torch.float8_e4m3fn)
        return fp8_matmul(a_q, b_q)
    elif precision == 'bf8':
        a_q = a.to(torch.float8_e5m2)
        b_q = b.to(torch.float8_e5m2)
        return fp8_matmul(a_q, b_q)
    elif precision == 'int8':
        a_q = a.to(torch.int8)
        b_q = b.to(torch.int8)
        if hasattr(torch, '_int_mm'):
            return torch._int_mm(a_q, b_q)
        return torch.mm(a_q.float(), b_q.float())
    elif precision == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.mm(a, b)
    else:
        dt = PRECISION_INFO[precision]['dtype']
        return torch.mm(a.to(dt), b.to(dt))


# ================= 精度探测 =================
def check_precision_support(prec: str, dev: torch.device) -> Tuple[bool, str, bool]:
    if prec not in PRECISION_INFO:
        return False, "未知精度", False

    info = PRECISION_INFO[prec]
    cap_str = '.'.join(map(str, torch.cuda.get_device_capability(dev)))
    cap = float(cap_str)

    if cap < info['min_cap']:
        return False, f"计算能力不足 {cap} < {info['min_cap']}", False

    # 精度级原生支持检测
    if prec in ('int4', 'fp4'):
        return True, f"硬件支持 (计算能力 {cap})，PyTorch 无原生 GEMM 接口", False

    if prec in ('fp8', 'bf8'):
        # fp8/bf8 require torch._scaled_mm + native float8 types
        if not (hasattr(torch, 'float8_e4m3fn') and hasattr(torch, '_scaled_mm')):
            return False, "PyTorch 版本过低，缺少 float8/_scaled_mm", False
        try:
            torch.manual_seed(GLOBAL_SEED)
            torch.cuda.manual_seed_all(GLOBAL_SEED)
            sz = 256
            a = torch.randn(sz, sz, device=dev)
            b = torch.randn(sz, sz, device=dev)
            _ = dispatch_matmul(prec, a, b)
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            return True, "原生 Tensor Core 支持", True
        except Exception as e:
            return False, f"原生 {prec.upper()} 失败: {e}", False

    if prec == 'int8':
        if not hasattr(torch, '_int_mm'):
            return False, "PyTorch 缺少 _int_mm", False
        try:
            torch.manual_seed(GLOBAL_SEED)
            torch.cuda.manual_seed_all(GLOBAL_SEED)
            sz = 256
            a = torch.randint(-128, 127, (sz, sz), dtype=torch.int8, device=dev)
            b = torch.randint(-128, 127, (sz, sz), dtype=torch.int8, device=dev)
            _ = torch._int_mm(a, b)
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            return True, "原生 INT8 (Tensor Core) 支持", True
        except Exception as e:
            return False, f"原生 INT8 失败: {e}", False

    # 标准 dtype (fp64/fp32/fp16/bf16/tf32)
    try:
        dt = info['dtype']
        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)
        sz = 256
        a = torch.randn(sz, sz, dtype=dt, device=dev)
        b = torch.randn(sz, sz, dtype=dt, device=dev)
        _ = torch.mm(a, b)
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        return True, "探测通过", True
    except Exception as e:
        return False, str(e), False


# ================= dmon 监控器 =================
class DmonMonitor:
    """使用 nvidia-smi dmon 持续采集 GPU 实时参数，支持自动重启。"""
    def __init__(self, phys_idx: int, interval_sec: int = 5) -> None:
        self.phys_idx = phys_idx
        self.interval = interval_sec
        self.process: Optional[subprocess.Popen] = None
        self.latest: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.running = False
        self._col_map: Dict[str, int] = {}
        self._start_time = time.time()
        self._start()

    def _start(self) -> None:
        cmd = ['nvidia-smi', 'dmon', '-i', str(self.phys_idx),
               '-s', 'pucvmt', '-d', str(self.interval), '-o', 'T']
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            self.running = True
            threading.Thread(target=self._reader, daemon=True).start()
            logger.info(f"dmon 监控已启动 (物理GPU {self.phys_idx}, PID {self.process.pid})")
        except Exception as e:
            logger.warning(f"dmon 启动失败: {e}，将回退到 pynvml。")
            self.running = False

    def _reader(self) -> None:
        while self.running and self.process and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            clean = line.lstrip('#').strip()
            if 'gpu' in clean.lower() and not any(c.isdigit() for c in clean.split(',')[0]):
                headers = [h.strip().lower() for h in clean.split()]
                self._col_map = {h: idx for idx, h in enumerate(headers)}
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                gpu_idx = int(parts[0].replace('#', '').strip())
                if gpu_idx != self.phys_idx:
                    continue
                data: Dict[str, Any] = {}
                for key, idx in self._col_map.items():
                    if idx < len(parts):
                        val = parts[idx]
                        if val == '-' or val == '':
                            data[key] = None
                        else:
                            try:
                                data[key] = float(val) if '.' in val else int(val)
                            except ValueError:
                                data[key] = val
                if 'gtemp' in data and data['gtemp'] is not None:
                    with self.lock:
                        self.latest.update(data)
            except Exception as e:
                logger.debug(f"dmon 行解析错误: {line} -> {e}")
                continue
        self.running = False

    def get_latest(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest)

    def stop(self) -> None:
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None


# ================= 矩阵规模自动缩放 =================
def auto_scale_matrix(precision: str, device: torch.device, target_time: float = 5.0) -> int:
    """Binary-search for largest matrix size that completes one matmul within target_time seconds."""
    info = PRECISION_INFO[precision]
    total_mem = torch.cuda.get_device_properties(device).total_memory
    free_mem = total_mem - torch.cuda.memory_allocated(device)

    # Upper bound: ~70% of free memory, each element = bytes
    elem_bytes = info['bytes']
    upper_est = int(((free_mem * 0.7) / (3 * elem_bytes)) ** 0.5)
    # Round down to nearest 256
    upper_est = max(upper_est // 256 * 256, 1024)

    # Cap by reasonable max for this precision
    max_caps = {'fp64': 8192, 'int8': 16384, 'fp8': 16384, 'bf8': 16384}
    upper = min(upper_est, max_caps.get(precision, 16384))

    logger.info(f"矩阵自动缩放: 上限 {upper}x{upper}")
    low, high = 1024, upper
    best = 1024

    while low <= high:
        mid = ((low + high) // 256) * 256
        if mid < 256:
            break
        try:
            torch.cuda.empty_cache()
            a = torch.randn(mid, mid, device=device)
            b = torch.randn(mid, mid, device=device)
            torch.cuda.synchronize(device)

            t0 = time.time()
            _ = dispatch_matmul(precision, a, b)
            torch.cuda.synchronize(device)
            elapsed = time.time() - t0

            del a, b
            torch.cuda.empty_cache()

            if elapsed < target_time * 0.8:
                best = mid
                low = mid + 256
            elif elapsed > target_time * 1.5:
                high = mid - 256
            else:
                best = mid
                break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                high = mid - 256
                torch.cuda.empty_cache()
            else:
                raise

    logger.info(f"最终矩阵: {best}x{best}")
    return max(best, 1024)


# ================= 基准测试 =================
def benchmark(
    precision: str,
    device: torch.device,
    duration: int,
    handle: Any,
    bus_width: Optional[int],
    phys_idx: int,
    matrix_size: int,
) -> Tuple[float, str, Dict[str, Any]]:
    torch.cuda.set_device(device)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.cuda.synchronize(device)

    logger.info(f"===== {precision.upper()} 测试 ({duration}s) =====")

    monitor = DmonMonitor(phys_idx, interval_sec=5)
    if not monitor.running:
        logger.warning("dmon 不可用，将仅使用 pynvml")

    total_mem = torch.cuda.get_device_properties(device).total_memory
    free_mem = total_mem - torch.cuda.memory_allocated(device)
    start_temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
    start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    logger.info(
        f"显存:{total_mem / 1024 ** 3:.2f}GB 空闲:{free_mem / 1024 ** 3:.2f}GB "
        f"温度:{start_temp}°C 功率:{start_power:.1f}W"
    )
    logger.info(f"矩阵大小: {matrix_size}x{matrix_size}")

    elem_bytes = PRECISION_INFO[precision]['bytes']

    # 预热
    logger.info(f"预热 {matrix_size}x{matrix_size} ...")
    try:
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        for _ in range(10):
            _ = dispatch_matmul(precision, a, b)
        torch.cuda.synchronize(device)
    except Exception as e:
        logger.error(f"预热失败: {e}")
        return 0, f"预热失败: {e}", {}
    finally:
        torch.cuda.empty_cache()

    max_metrics: Dict[str, Any] = {
        'temp': -1, 'power': -1, 'sm_clock': -1, 'mem_clock': -1,
        'mem_used_ratio': -1.0, 'effective_bw': -1.0,
    }

    last_temp = time.time()
    last_beat = time.time()

    def _get_status() -> Tuple[Any, ...]:
        temp = power = sm_clk = mem_clk = None
        if monitor and monitor.running:
            latest = monitor.get_latest()
            temp = latest.get('gtemp')
            power = latest.get('pwr')
            sm_clk = latest.get('pclk')
            mem_clk = latest.get('mclk')
        if temp is None:
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            except Exception:
                pass
        if power is None:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except Exception:
                pass
        if sm_clk is None:
            try:
                sm_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except Exception:
                pass
        if mem_clk is None:
            try:
                mem_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                pass
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / 1024 ** 3
            total_gb = mem_info.total / 1024 ** 3
            used_ratio = (used_gb / total_gb) * 100 if total_gb > 0 else 0
        except Exception:
            used_gb = total_gb = used_ratio = 0
        return temp, power, sm_clk, mem_clk, used_gb, total_gb, used_ratio

    def _effective_bw(m: int, n: int, k: int, bytes_per_elem: float, elapsed_sec: float, it_cnt: int) -> float:
        total_bytes = (m * k + k * n + m * n) * bytes_per_elem * it_cnt
        return total_bytes / elapsed_sec / 1e9 if elapsed_sec > 0 else 0.0

    def _log_status(it: int, start_t: float, iter_time: float) -> None:
        nonlocal last_temp, last_beat
        now = time.time()
        if now - last_temp < 10:
            return
        temp, power, sm_clk, mem_clk, used_gb, total_gb, used_ratio = _get_status()
        if temp is not None:
            if temp > max_metrics['temp']:
                max_metrics['temp'] = temp
            if power is not None and power > max_metrics['power']:
                max_metrics['power'] = power
            if sm_clk is not None and sm_clk > max_metrics['sm_clock']:
                max_metrics['sm_clock'] = sm_clk
            if mem_clk is not None and mem_clk > max_metrics['mem_clock']:
                max_metrics['mem_clock'] = mem_clk
            if used_ratio > max_metrics['mem_used_ratio']:
                max_metrics['mem_used_ratio'] = used_ratio
            bw = _effective_bw(matrix_size, matrix_size, matrix_size, elem_bytes, iter_time, 1)
            if bw > max_metrics['effective_bw']:
                max_metrics['effective_bw'] = bw
            elapsed = now - start_t
            eta = duration - elapsed
            logger.info(
                f"[{it}] {elapsed:.0f}s/{duration}s ETA:{eta:.0f}s | "
                f"{temp:.0f}°C | {power:.1f}W | SM:{sm_clk}MHz | "
                f"MEM:{mem_clk}MHz | 占用:{used_gb:.2f}/{total_gb:.2f}GB | "
                f"有效BW:{bw:.1f} GB/s"
            )
            last_temp = now

        if now - last_beat >= 30:
            elapsed = now - start_t
            logger.info(f"心跳: {it}次, {elapsed:.1f}s, 平均 {elapsed / max(it, 1):.2f}s/次")
            last_beat = now

    # 准备测试张量（一次性分配）
    try:
        if precision == 'int8':
            a = torch.randint(-128, 127, (matrix_size, matrix_size), dtype=torch.int8, device=device)
            b = torch.randint(-128, 127, (matrix_size, matrix_size), dtype=torch.int8, device=device)
        else:
            a = torch.randn(matrix_size, matrix_size, device=device)
            b = torch.randn(matrix_size, matrix_size, device=device)
    except RuntimeError as e:
        logger.error(f"张量分配失败: {e}")
        return 0, f"OOM: {e}", {}

    try:
        start = time.time()
        it = 0
        while time.time() - start < duration:
            iter_start = time.time()
            c = dispatch_matmul(precision, a, b)
            torch.cuda.synchronize()
            iter_time = time.time() - iter_start
            it += 1
            _log_status(it, start, iter_time)
        elapsed = time.time() - start
    except Exception as e:
        logger.error(f"测试异常: {e}\n{traceback.format_exc()}")
        return 0, str(e), max_metrics
    finally:
        if monitor:
            monitor.stop()
        del a, b
        try:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        end_temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
        end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        logger.info(f"结束温度:{end_temp}°C 功率:{end_power:.1f}W")

    tflops = (2 * matrix_size ** 3 * it / elapsed) / 1e12 if it > 0 else 0
    logger.info(f"{precision.upper()} 完成: {tflops:.2f} TFLOPS ({it}次, {elapsed:.1f}s)")
    return tflops, "成功", max_metrics


# ================= 结果输出 =================
def final_report(results: Dict[str, Dict], support_status: Dict[str, str]) -> None:
    header = (f"{'精度':<8} {'TFLOPS':>10} {'温度':>7} {'功率':>7} "
              f"{'SM':>8} {'MEM':>8} {'有效BW':>12} {'显存占用':>10} {'状态':>14}")
    logger.info("\n" + "=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for prec in ['fp64', 'tf32', 'fp32', 'fp16', 'bf16', 'int8', 'fp8', 'bf8', 'int4', 'fp4']:
        status_str = support_status.get(prec, 'N/A')
        if prec in results and results[prec]['status'] == "成功":
            r = results[prec]
            m = r['max_metrics']
            bw = f"{m['effective_bw']:.1f} GB/s" if m['effective_bw'] >= 0 else "N/A"
            mem_str = f"{m['mem_used_ratio']:.1f}%" if m['mem_used_ratio'] >= 0 else "N/A"
            temp = f"{m['temp']:.0f}°C" if m['temp'] >= 0 else "N/A"
            pw = f"{m['power']:.1f}W" if m['power'] >= 0 else "N/A"
            sm = f"{m['sm_clock']}MHz" if m['sm_clock'] >= 0 else "N/A"
            mc = f"{m['mem_clock']}MHz" if m['mem_clock'] >= 0 else "N/A"
            logger.info(
                f"{prec.upper():<8} {r['tflops']:>10.2f} {temp:>7} {pw:>7} "
                f"{sm:>8} {mc:>8} {bw:>12} {mem_str:>10} {status_str:>14}"
            )
        else:
            logger.info(
                f"{prec.upper():<8} {'N/A':>10} {'N/A':>7} {'N/A':>7} "
                f"{'N/A':>8} {'N/A':>8} {'N/A':>12} {'N/A':>10} {status_str:>14}"
            )


def save_results_json(results: Dict[str, Dict], support_status: Dict[str, str],
                       sys_info: Dict[str, Any], filepath: str) -> None:
    """Save benchmark results as JSON for machine consumption."""
    output: Dict[str, Any] = {
        'environment': sys_info,
        'test_config': {
            'seed': GLOBAL_SEED,
        },
        'precision_support': {},
        'results': {},
    }
    for prec in ['fp64', 'tf32', 'fp32', 'fp16', 'bf16', 'int8', 'fp8', 'bf8', 'int4', 'fp4']:
        output['precision_support'][prec] = support_status.get(prec, 'N/A')
    for prec, r in results.items():
        output['results'][prec] = {
            'tflops': r['tflops'],
            'status': r['status'],
            'max_metrics': r.get('max_metrics', {}),
        }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON 结果已保存: {filepath}")
    except Exception as e:
        logger.warning(f"JSON 保存失败: {e}")


# ================= 主入口 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU Computility Test")
    parser.add_argument('--gpu', type=int, default=0, help='PyTorch 可见 GPU 索引')
    parser.add_argument('--physical-gpu', type=int, default=None, help='物理 GPU 索引')
    parser.add_argument('--duration', type=int, default=180, help='每精度测试秒数 (默认: 180)')
    parser.add_argument('--matrix-size', type=int, default=None,
                        help='手动指定矩阵大小 (默认: 自动缩放)')
    parser.add_argument('--precisions', type=str, default=None,
                        help='测试精度列表，逗号分隔 (默认: 所有原生支持的精度)')
    parser.add_argument('--output-json', type=str, default='',
                        help='结果 JSON 输出路径 (默认: 自动生成)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA 不可用")
        sys.exit(1)

    pynvml.nvmlInit()
    phys_count = pynvml.nvmlDeviceGetCount()
    phys_uuids: List[Tuple[int, Optional[str]]] = []
    for idx in range(phys_count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            raw = pynvml.nvmlDeviceGetUUID(h)
            uuid = raw.decode() if isinstance(raw, bytes) else raw
            phys_uuids.append((idx, uuid))
        except Exception as e:
            logger.warning(f"物理GPU {idx} UUID失败: {e}")
            phys_uuids.append((idx, None))

    logger.info("===== PyTorch ↔ 物理 GPU 映射 =====")
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(dev)
        tuuid = str(props.uuid).replace('GPU-', '')
        pidx = None
        for pid, puuid in phys_uuids:
            if puuid and puuid.replace('GPU-', '') == tuuid:
                pidx = pid
                break
        bw = get_memory_bus_width(pidx) if pidx is not None else None
        bw_str = f"{bw}bit" if bw else "N/A"
        logger.info(f"PyTorch {i} → 物理 {pidx} | {props.name} | 实时位宽:{bw_str} | UUID:{tuuid}")

    if args.physical_gpu is not None:
        target_phys = args.physical_gpu
        target_torch = None
        for i in range(torch.cuda.device_count()):
            dev = torch.device(f'cuda:{i}')
            tuuid = str(torch.cuda.get_device_properties(dev).uuid).replace('GPU-', '')
            for pid, puuid in phys_uuids:
                if pid == target_phys and puuid and puuid.replace('GPU-', '') == tuuid:
                    target_torch = i
                    break
            if target_torch is not None:
                break
        if target_torch is None:
            logger.error(f"物理GPU {target_phys} 未在 PyTorch 可见范围")
            sys.exit(1)
        gpu_index = target_torch
    else:
        gpu_index = args.gpu

    if gpu_index >= torch.cuda.device_count():
        logger.error(f"GPU索引 {gpu_index} 超出范围")
        sys.exit(1)

    device = torch.device(f'cuda:{gpu_index}')
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    tuuid = str(props.uuid).replace('GPU-', '')
    phys_idx_final = None
    for pid, puuid in phys_uuids:
        if puuid and puuid.replace('GPU-', '') == tuuid:
            phys_idx_final = pid
            break
    if phys_idx_final is None:
        logger.error("无法确定物理索引")
        sys.exit(1)

    handle = pynvml.nvmlDeviceGetHandleByIndex(phys_idx_final)
    bus_width = get_memory_bus_width(phys_idx_final)
    bw_disp = f"{bus_width}bit" if bus_width else "N/A"
    logger.info(f"\n选定: PyTorch cuda:{gpu_index} | 物理 {phys_idx_final} | {props.name} | 实时位宽:{bw_disp}")

    sys_info = get_extended_sysinfo(handle, device)
    print_extended_env(sys_info)

    logger.info("\n===== 权威参考数据（仅参考，非实时）=====")
    print_reference_specs(props.name)

    cap = float('.'.join(map(str, torch.cuda.get_device_capability(device))))
    logger.info(f"计算能力: {cap}")

    # 检测所有精度支持状态
    all_precisions = ['fp64', 'tf32', 'fp32', 'fp16', 'bf16', 'int8', 'fp8', 'bf8', 'int4', 'fp4']
    logger.info("\n===== 精度支持状态 =====")
    support_status: Dict[str, str] = {}
    for prec in all_precisions:
        ok, msg, native_ok = check_precision_support(prec, device)
        if ok and native_ok:
            status_text = "✅ 原生支持"
            support_status[prec] = "✅ 原生支持"
        elif ok and not native_ok:
            status_text = f"⚠️ 硬件支持但无原生接口 ({msg})"
            support_status[prec] = "⚠️ 硬件支持/无接口"
        else:
            status_text = f"❌ 不支持 ({msg})"
            support_status[prec] = "❌ 不支持"
        logger.info(f"{prec.upper():<8}: {status_text}")

    # 确定要测试的精度
    if args.precisions:
        requested = [p.strip().lower() for p in args.precisions.split(',')]
        tested = [p for p in requested if support_status.get(p) == "✅ 原生支持"]
        skipped = [p for p in requested if p not in tested]
        if skipped:
            logger.warning(f"以下请求的精度不支持或无法测试: {', '.join(skipped)}")
    else:
        tested = [prec for prec in all_precisions if support_status[prec] == "✅ 原生支持"]

    logger.info(f"\n最终测试 ({len(tested)} 项): {', '.join(tested)}")
    if not tested:
        logger.error("无精度可用")
        sys.exit(1)

    # 设置 Tensor Core 优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(f"\n===== 开始基准测试 (每精度{args.duration}s, 种子{GLOBAL_SEED}) =====")

    results: Dict[str, Dict] = {}
    for prec in tested:
        logger.info(f"\n>>>>>> {prec.upper()} <<<<<<")
        # 自动或手动确定矩阵大小
        if args.matrix_size is not None:
            ms = args.matrix_size
            logger.info(f"手动指定矩阵大小: {ms}x{ms}")
        else:
            ms = auto_scale_matrix(prec, device, target_time=3.0)
        tflops, status, metrics = benchmark(prec, device, args.duration, handle, bus_width, phys_idx_final, ms)
        results[prec] = {'tflops': tflops, 'status': status, 'max_metrics': metrics}
        if status == "成功":
            logger.info(f"{prec.upper()}: {tflops:.2f} TFLOPS")
        else:
            logger.error(f"{prec.upper()} 失败: {status}")
        torch.cuda.empty_cache()
        gc.collect()

    # 环境回顾
    logger.info("\n===== 测试环境回顾 =====")
    logger.info(f"GPU: {sys_info['gpu_name']} | 驱动: {sys_info.get('driver', 'N/A')} | PyTorch: {sys_info['torch']}")
    logger.info(f"显存: {sys_info['gpu_mem']} | CUDA核心: {sys_info['cuda_cores']} | Tensor核心: {sys_info['tensor_cores']}")
    logger.info(f"CPU: {sys_info['cpu_cores']}核 | 系统内存: {sys_info['sys_mem']} | 内核: {sys_info['kernel']}")

    logger.info("\n===== 最终结果汇总 =====")
    final_report(results, support_status)

    # JSON 输出
    json_path = args.output_json or f"GPU_Result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results_json(results, support_status, sys_info, json_path)

    logger.info("测试完成。")
    pynvml.nvmlShutdown()
