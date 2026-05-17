"""
DE (Development Ephemeris) 切比雪夫插值模块

NASA JPL 的 DE 系列星历表使用切比雪夫多项式插值来存储和计算
行星位置。该模块实现：
- 切比雪夫多项式计算
- 二进制星历数据读取
- 高阶插值计算

切比雪夫多项式公式:
    x(t) = Σ(i=0 to n) c_i · T_i(τ)

其中 T_i(τ) 是第 i 阶切比雪夫多项式，τ 是归一化时间。
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, BinaryIO
import struct


class ChebyshevPolynomial:
    """
    切比雪夫多项式计算器
    
    切比雪夫多项式定义:
    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x·T_{n-1}(x) - T_{n-2}(x)  (n ≥ 2)
    """
    
    @staticmethod
    def evaluate(tau: float, n: int) -> np.ndarray:
        """
        计算 0 到 n 阶的切比雪夫多项式值
        
        Args:
            tau: 归一化时间，范围 [-1, 1]
            n: 最高阶数
        
        Returns:
            数组 [T_0(tau), T_1(tau), ..., T_n(tau)]
        """
        T = np.zeros(n + 1)
        T[0] = 1.0
        if n >= 1:
            T[1] = tau
        
        # 递推计算
        for i in range(2, n + 1):
            T[i] = 2.0 * tau * T[i - 1] - T[i - 2]
        
        return T
    
    @staticmethod
    def evaluate_batch(tau_array: np.ndarray, n: int) -> np.ndarray:
        """
        批量计算切比雪夫多项式值（向量化）
        
        Args:
            tau_array: 归一化时间数组，范围 [-1, 1]，形状 (m,)
            n: 最高阶数
        
        Returns:
            数组形状 (m, n+1)，每行是 [T_0(tau), T_1(tau), ..., T_n(tau)]
        """
        m = len(tau_array)
        T = np.zeros((m, n + 1))
        T[:, 0] = 1.0
        if n >= 1:
            T[:, 1] = tau_array
        
        # 递推计算
        for i in range(2, n + 1):
            T[:, i] = 2.0 * tau_array * T[:, i - 1] - T[:, i - 2]
        
        return T
    
    @staticmethod
    def derivative(tau: float, n: int) -> np.ndarray:
        """
        计算切比雪夫多项式的导数
        
        dT_0/dx = 0
        dT_1/dx = 1
        dT_n/dx = 2·T_{n-1} + 2x·dT_{n-1}/dx - dT_{n-2}/dx
        
        Args:
            tau: 归一化时间
            n: 最高阶数
        
        Returns:
            导数数组 [T'_0(tau), T'_1(tau), ..., T'_n(tau)]
        """
        T = ChebyshevPolynomial.evaluate(tau, n)
        dT = np.zeros(n + 1)
        
        if n >= 1:
            dT[1] = 1.0
        
        for i in range(2, n + 1):
            dT[i] = 2.0 * T[i - 1] + 2.0 * tau * dT[i - 1] - dT[i - 2]
        
        return dT


class DEInterpolator:
    """
    DE 星历表切比雪夫插值器
    
    使用切比雪夫多项式进行高阶插值：
        x(t) = Σ(i=0 to n) c_i · T_i(τ)
    
    其中 τ = 2(t - t_mid) / Δt，将时间归一化到 [-1, 1]
    """
    
    def __init__(self, coefficients: np.ndarray, t_start: float, t_end: float):
        """
        初始化插值器
        
        Args:
            coefficients: 切比雪夫系数数组 [c_0, c_1, ..., c_n]
            t_start: 时间段起始时间
            t_end: 时间段结束时间
        """
        self.coefficients = np.array(coefficients)
        self.n = len(coefficients) - 1  # 最高阶数
        self.t_start = t_start
        self.t_end = t_end
        self.t_mid = (t_start + t_end) / 2.0
        self.dt = t_end - t_start
    
    def _normalize_time(self, t: float) -> float:
        """
        将时间归一化到 [-1, 1]
        
        Args:
            t: 实际时间
        
        Returns:
            归一化时间 tau
        """
        return 2.0 * (t - self.t_mid) / self.dt
    
    def interpolate(self, t: float) -> float:
        """
        在时刻 t 进行插值计算
        
        Args:
            t: 目标时刻
        
        Returns:
            插值结果 x(t)
        """
        tau = self._normalize_time(t)
        
        # 计算切比雪夫多项式值
        T = ChebyshevPolynomial.evaluate(tau, self.n)
        
        # 计算插值结果: x(t) = Σ c_i · T_i(tau)
        result = np.dot(self.coefficients, T)
        
        return result
    
    def interpolate_derivative(self, t: float) -> float:
        """
        计算插值函数的导数 dx/dt
        
        Args:
            t: 目标时刻
        
        Returns:
            dx/dt
        """
        tau = self._normalize_time(t)
        
        # 计算切比雪夫多项式导数
        dT = ChebyshevPolynomial.derivative(tau, self.n)
        
        # dx/dt = dx/dtau · dtau/dt = (Σ c_i · T'_i(tau)) · (2/Δt)
        dx_dtau = np.dot(self.coefficients, dT)
        dx_dt = dx_dtau * (2.0 / self.dt)
        
        return dx_dt
    
    def interpolate_batch(self, t_array: np.ndarray) -> np.ndarray:
        """
        批量插值计算（向量化）
        
        Args:
            t_array: 时间数组，形状 (m,)
        
        Returns:
            插值结果数组，形状 (m,)
        """
        # 归一化时间
        tau_array = 2.0 * (t_array - self.t_mid) / self.dt
        
        # 批量计算切比雪夫多项式
        T_matrix = ChebyshevPolynomial.evaluate_batch(tau_array, self.n)
        
        # 向量化矩阵乘法: (m, n+1) @ (n+1,) -> (m,)
        results = T_matrix @ self.coefficients
        
        return results


class DEBinaryReader:
    """
    DE 星历表二进制文件读取器
    
    JPL DE 星历表通常以二进制格式存储，包含：
    - 文件头信息
    - 切比雪夫系数数据
    """
    
    def __init__(self, file_path: str):
        """
        初始化读取器
        
        Args:
            file_path: DE 二进制文件路径
        """
        self.file_path = Path(file_path)
        self.header = {}
        self.coefficients = None
        
    def read_header(self) -> dict:
        """
        读取文件头信息
        
        DE 文件头通常包含：
        - 起始和结束儒略日
        - 行星数量
        - 每个行星的子区间数
        - 每个子区间的切比雪夫系数数量
        - 每个子区间的天数
        
        Returns:
            头信息字典
        """
        with open(self.file_path, 'rb') as f:
            # 读取头记录（通常是两个 84 字节的记录）
            # 注意：实际格式取决于具体的 DE 版本
            
            # 读取起始和结束儒略日（双精度浮点数）
            f.seek(0)
            data = f.read(8)
            if len(data) == 8:
                self.header['jd_start'] = struct.unpack('d', data)[0]
            
            data = f.read(8)
            if len(data) == 8:
                self.header['jd_end'] = struct.unpack('d', data)[0]
            
            # 读取其他头信息...
            # 这里简化处理，实际应根据具体 DE 版本解析
            
        return self.header
    
    def read_coefficients(self, offset: int = 0, count: int = None) -> np.ndarray:
        """
        读取切比雪夫系数
        
        Args:
            offset: 起始偏移量（以系数个数为单位）
            count: 读取的系数个数，None 表示读取所有
        
        Returns:
            系数数组
        """
        with open(self.file_path, 'rb') as f:
            # 跳过文件头（假设头大小为 0，实际应根据 DE 版本调整）
            f.seek(offset * 8)  # 每个系数是 8 字节的双精度浮点数
            
            if count is None:
                # 读取到文件末尾
                data = f.read()
            else:
                data = f.read(count * 8)
            
            # 将二进制数据转换为双精度浮点数数组
            n_coeffs = len(data) // 8
            coefficients = struct.unpack(f'{n_coeffs}d', data[:n_coeffs * 8])
            
            return np.array(coefficients)
    
    def parse_file(self) -> List[DEInterpolator]:
        """
        解析整个 DE 文件，返回插值器列表
        
        Returns:
            DEInterpolator 列表，每个对应一个时间子区间
        """
        # 这里需要根据具体的 DE 文件格式实现
        # 简化版本：假设文件只包含系数和基本时间信息
        
        interpolators = []
        
        # 读取头信息
        self.read_header()
        
        # 读取所有系数
        all_coeffs = self.read_coefficients()
        
        # 根据 DE 文件结构分割系数
        # 这里使用简化的假设：每 13 个系数为一个子区间
        coeffs_per_segment = 13
        n_segments = len(all_coeffs) // coeffs_per_segment
        
        jd_start = self.header.get('jd_start', 2451545.0)
        jd_end = self.header.get('jd_end', 2451545.0 + 32.0)
        
        segment_duration = (jd_end - jd_start) / n_segments
        
        for i in range(n_segments):
            start_idx = i * coeffs_per_segment
            end_idx = start_idx + coeffs_per_segment
            coeffs = all_coeffs[start_idx:end_idx]
            
            t_start = jd_start + i * segment_duration
            t_end = t_start + segment_duration
            
            interpolator = DEInterpolator(coeffs, t_start, t_end)
            interpolators.append(interpolator)
        
        return interpolators


class DESimpleData:
    """
    简化的 DE 数据类，用于演示切比雪夫插值
    
    使用模拟数据展示插值功能
    """
    
    @staticmethod
    def generate_sample_data(n_segments: int = 4, 
                            coeffs_per_segment: int = 13,
                            jd_start: float = 2451545.0,
                            jd_end: float = 2451545.0 + 32.0) -> List[DEInterpolator]:
        """
        生成示例 DE 插值数据
        
        模拟地球位置的切比雪夫系数
        
        Args:
            n_segments: 时间子区间数量
            coeffs_per_segment: 每个子区间的系数数量
            jd_start: 起始儒略日
            jd_end: 结束儒略日
        
        Returns:
            DEInterpolator 列表
        """
        interpolators = []
        segment_duration = (jd_end - jd_start) / n_segments
        
        np.random.seed(42)  # 保证可重复
        
        for i in range(n_segments):
            # 生成模拟系数（实际应从 DE 文件读取）
            # 系数通常随阶数增加而衰减
            coeffs = np.zeros(coeffs_per_segment)
            coeffs[0] = 1.0  # 常数项
            coeffs[1] = 0.1  # 线性项
            for j in range(2, coeffs_per_segment):
                coeffs[j] = 0.01 / (j ** 2)  # 高阶项衰减
            
            # 添加一些随机扰动模拟真实数据
            coeffs += np.random.normal(0, 0.001, coeffs_per_segment)
            
            t_start = jd_start + i * segment_duration
            t_end = t_start + segment_duration
            
            interpolator = DEInterpolator(coeffs, t_start, t_end)
            interpolators.append(interpolator)
        
        return interpolators


def demo():
    """演示 DE 切比雪夫插值"""
    print("=== DE 切比雪夫插值演示 ===\n")
    
    # 1. 切比雪夫多项式计算演示
    print("1. 切比雪夫多项式值:")
    tau = 0.5
    T = ChebyshevPolynomial.evaluate(tau, 5)
    for i, t in enumerate(T):
        print(f"   T_{i}({tau}) = {t:.6f}")
    
    # 2. 生成示例插值数据
    print("\n2. 生成示例 DE 插值数据:")
    interpolators = DESimpleData.generate_sample_data(
        n_segments=4,
        coeffs_per_segment=13,
        jd_start=2451545.0,
        jd_end=2451545.0 + 32.0
    )
    print(f"   生成了 {len(interpolators)} 个插值器")
    print(f"   每个插值器使用 {interpolators[0].n + 1} 个切比雪夫系数")
    
    # 3. 单点插值
    print("\n3. 单点插值计算:")
    test_jd = 2451545.0 + 8.0  # 第2个子区间中间
    
    # 找到对应的插值器
    for interp in interpolators:
        if interp.t_start <= test_jd <= interp.t_end:
            result = interp.interpolate(test_jd)
            derivative = interp.interpolate_derivative(test_jd)
            print(f"   JD = {test_jd}")
            print(f"   插值结果 x(t) = {result:.6f}")
            print(f"   导数 dx/dt = {derivative:.8f}")
            break
    
    # 4. 批量插值
    print("\n4. 批量插值计算:")
    test_jds = np.linspace(2451545.0, 2451545.0 + 32.0, 100)
    
    results = []
    for jd in test_jds:
        for interp in interpolators:
            if interp.t_start <= jd <= interp.t_end:
                results.append(interp.interpolate(jd))
                break
    
    results = np.array(results)
    print(f"   计算了 {len(results)} 个时间点的插值")
    print(f"   结果范围: [{results.min():.6f}, {results.max():.6f}]")
    print(f"   结果均值: {results.mean():.6f}")
    
    # 5. 向量化批量插值
    print("\n5. 向量化批量插值（单个子区间）:")
    interp = interpolators[0]
    t_batch = np.linspace(interp.t_start, interp.t_end, 50)
    results_batch = interp.interpolate_batch(t_batch)
    print(f"   在 [{t_batch[0]:.1f}, {t_batch[-1]:.1f}] 区间内计算了 {len(results_batch)} 个点")
    print(f"   前5个结果: {results_batch[:5]}")
    
    # 6. 插值精度分析
    print("\n6. 插值精度分析:")
    # 用高分辨率采样检查连续性
    continuity_errors = []
    for i in range(len(interpolators) - 1):
        # 在子区间边界处检查连续性
        boundary = interpolators[i].t_end
        left_value = interpolators[i].interpolate(boundary)
        right_value = interpolators[i + 1].interpolate(boundary)
        error = abs(left_value - right_value)
        continuity_errors.append(error)
    
    if continuity_errors:
        print(f"   子区间边界最大不连续误差: {max(continuity_errors):.2e}")
        print(f"   子区间边界平均不连续误差: {np.mean(continuity_errors):.2e}")


if __name__ == "__main__":
    demo()
