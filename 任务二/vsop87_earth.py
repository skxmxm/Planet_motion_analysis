"""
VSOP87 地球坐标计算模块

VSOP87 (Variations Seculaires des Orbites Planetaires)
是法国天体力学与星历计算研究所发布的行星分析理论。
该模块实现地球坐标的计算，包括：
- 高效的系数表 Parser
- NumPy 向量化运算
- 日心黄道坐标计算

参考: http://www.neoprogrammics.com/vsop87/
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re


@dataclass
class VSOP87Term:
    """VSOP87 级数展开中的单个项"""
    rank: int           # 项的序号
    amplitude: float    # 振幅 A
    phase: float        # 相位 B
    frequency: float    # 频率 C


@dataclass
class VSOP87Variable:
    """VSOP87 变量（如黄经、纬度、距离等）"""
    variable_index: int         # 变量索引 (1-6)
    exponent: int               # 幂次
    terms: List[VSOP87Term]     # 级数项列表


class VSOP87Parser:
    """
    VSOP87 系数表的高效解析器
    
    数据格式（每行）:
    变量索引  幂次  项序号  振幅(A)  相位(B)  频率(C)
    """
    
    def __init__(self, data_path: str):
        """
        初始化解析器
        
        Args:
            data_path: VSOP87 数据文件路径
        """
        self.data_path = Path(data_path)
        self.variables: List[VSOP87Variable] = []
        
    def parse(self) -> List[VSOP87Variable]:
        """
        解析 VSOP87 系数文件
        
        Returns:
            变量列表，每个变量包含其级数展开项
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"VSOP87 数据文件未找到: {self.data_path}")
        
        current_terms = []
        current_var_idx = None
        current_exponent = None
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 6:
                    continue
                
                var_idx = int(parts[0])
                exponent = int(parts[1])
                rank = int(parts[2])
                amplitude = float(parts[3])
                phase = float(parts[4])
                frequency = float(parts[5])
                
                # 如果遇到新的变量或幂次，保存之前的
                if (current_var_idx is not None and 
                    (var_idx != current_var_idx or exponent != current_exponent)):
                    self.variables.append(VSOP87Variable(
                        variable_index=current_var_idx,
                        exponent=current_exponent,
                        terms=current_terms
                    ))
                    current_terms = []
                
                current_var_idx = var_idx
                current_exponent = exponent
                current_terms.append(VSOP87Term(
                    rank=rank,
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency
                ))
        
        # 保存最后一个变量
        if current_terms:
            self.variables.append(VSOP87Variable(
                variable_index=current_var_idx,
                exponent=current_exponent,
                terms=current_terms
            ))
        
        return self.variables
    
    def parse_numpy(self) -> dict:
        """
        使用 NumPy 进行高效解析，返回向量化数据结构
        
        Returns:
            字典，键为 (variable_index, exponent)，值为 NumPy 数组
            数组形状: (n_terms, 3)，列分别为 [amplitude, phase, frequency]
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"VSOP87 数据文件未找到: {self.data_path}")
        
        data_dict = {}
        
        # 读取所有行并解析
        raw_data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    raw_data.append([
                        int(parts[0]),      # var_idx
                        int(parts[1]),      # exponent
                        float(parts[3]),    # amplitude
                        float(parts[4]),    # phase
                        float(parts[5])     # frequency
                    ])
        
        if not raw_data:
            return data_dict
        
        # 转换为 NumPy 数组
        data_array = np.array(raw_data)
        
        # 按变量索引和幂次分组
        var_indices = np.unique(data_array[:, 0])
        
        for var_idx in var_indices:
            mask_var = data_array[:, 0] == var_idx
            var_data = data_array[mask_var]
            
            exponents = np.unique(var_data[:, 1])
            for exp in exponents:
                mask_exp = var_data[:, 1] == exp
                terms_data = var_data[mask_exp, 2:]  # [amplitude, phase, frequency]
                data_dict[(int(var_idx), int(exp))] = terms_data
        
        return data_dict


class VSOP87Earth:
    """
    VSOP87 地球坐标计算器
    
    计算地球的日心黄道坐标：
    - L: 黄经 (Longitude)
    - B: 黄纬 (Latitude)  
    - R: 距离 (Radius)
    """
    
    # VSOP87 变量定义
    VARIABLES = {
        'L0': (1, 0),  # 黄经，幂次 0
        'L1': (1, 1),  # 黄经，幂次 1
        'L2': (1, 2),  # 黄经，幂次 2
        'L3': (1, 3),  # 黄经，幂次 3
        'L4': (1, 4),  # 黄经，幂次 4
        'L5': (1, 5),  # 黄经，幂次 5
        'B0': (2, 0),  # 黄纬，幂次 0
        'B1': (2, 1),  # 黄纬，幂次 1
        'B2': (2, 2),  # 黄纬，幂次 2
        'B3': (2, 3),  # 黄纬，幂次 3
        'B4': (2, 4),  # 黄纬，幂次 4
        'B5': (2, 5),  # 黄纬，幂次 5
        'R0': (3, 0),  # 距离，幂次 0
        'R1': (3, 1),  # 距离，幂次 1
        'R2': (3, 2),  # 距离，幂次 2
        'R3': (3, 3),  # 距离，幂次 3
        'R4': (3, 4),  # 距离，幂次 4
        'R5': (3, 5),  # 距离，幂次 5
    }
    
    def __init__(self, data_path: str):
        """
        初始化地球坐标计算器
        
        Args:
            data_path: VSOP87 地球数据文件路径
        """
        self.parser = VSOP87Parser(data_path)
        self.data = self.parser.parse_numpy()
        
        # 预计算各变量的 NumPy 数组以便向量化运算
        self._precompute_arrays()
    
    def _precompute_arrays(self):
        """预计算数组，加速后续计算"""
        self.var_arrays = {}
        for name, key in self.VARIABLES.items():
            if key in self.data:
                self.var_arrays[name] = self.data[key]
    
    def _compute_variable(self, var_name: str, t: float) -> float:
        """
        计算单个变量的值
        
        Args:
            var_name: 变量名（如 'L0', 'L1' 等）
            t: 儒略千年数 (Julian Millennia from J2000.0)
        
        Returns:
            变量的值
        """
        if var_name not in self.var_arrays:
            return 0.0
        
        terms = self.var_arrays[var_name]
        
        # 向量化计算: sum(A * cos(B + C * t))
        # terms[:, 0] = amplitude (A)
        # terms[:, 1] = phase (B)
        # terms[:, 2] = frequency (C)
        
        amplitudes = terms[:, 0]
        phases = terms[:, 1]
        frequencies = terms[:, 2]
        
        # 向量化计算所有项
        values = amplitudes * np.cos(phases + frequencies * t)
        
        return np.sum(values)
    
    def compute_coordinates(self, jd: float) -> Tuple[float, float, float]:
        """
        计算给定儒略日的地球日心黄道坐标
        
        Args:
            jd: 儒略日 (Julian Date)
        
        Returns:
            (L, B, R) 元组:
            - L: 黄经 (弧度)
            - B: 黄纬 (弧度)
            - R: 距离 (AU)
        """
        # 转换为儒略千年数（从 J2000.0 起算）
        # J2000.0 = 2451545.0
        t = (jd - 2451545.0) / 365250.0
        
        # 计算黄经 L = L0 + L1*t + L2*t^2 + L3*t^3 + L4*t^4 + L5*t^5
        L = sum(
            self._compute_variable(f'L{i}', t) * (t ** i)
            for i in range(6)
        )
        
        # 计算黄纬 B = B0 + B1*t + B2*t^2 + B3*t^3 + B4*t^4 + B5*t^5
        B = sum(
            self._compute_variable(f'B{i}', t) * (t ** i)
            for i in range(6)
        )
        
        # 计算距离 R = R0 + R1*t + R2*t^2 + R3*t^3 + R4*t^4 + R5*t^5
        R = sum(
            self._compute_variable(f'R{i}', t) * (t ** i)
            for i in range(6)
        )
        
        # 规范化角度到 [0, 2π]
        L = L % (2 * np.pi)
        B = B % (2 * np.pi)
        
        return L, B, R
    
    def compute_batch(self, jds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量计算多个儒略日的地球坐标（完全向量化）
        
        Args:
            jds: 儒略日数组，形状 (n,)
        
        Returns:
            (L_array, B_array, R_array)，每个数组形状 (n,)
        """
        # 转换为儒略千年数
        t = (jds - 2451545.0) / 365250.0
        n = len(t)
        
        L = np.zeros(n)
        B = np.zeros(n)
        R = np.zeros(n)
        
        # 对每个幂次进行计算
        for i in range(6):
            t_power = t ** i
            
            # 黄经
            var_name = f'L{i}'
            if var_name in self.var_arrays:
                terms = self.var_arrays[var_name]
                # 向量化: (n_terms,) @ (n_terms, n) -> (n,)
                # 使用广播: cos(B + C * t[:, None])
                phases = terms[:, 1:2]  # (n_terms, 1)
                freqs = terms[:, 2:3]   # (n_terms, 1)
                amps = terms[:, 0]      # (n_terms,)
                
                # (n_terms, n)
                cos_vals = np.cos(phases + freqs * t[None, :])
                # (n_terms,) @ (n_terms, n) -> (n,)
                L += t_power * (amps @ cos_vals)
            
            # 黄纬
            var_name = f'B{i}'
            if var_name in self.var_arrays:
                terms = self.var_arrays[var_name]
                phases = terms[:, 1:2]
                freqs = terms[:, 2:3]
                amps = terms[:, 0]
                
                cos_vals = np.cos(phases + freqs * t[None, :])
                B += t_power * (amps @ cos_vals)
            
            # 距离
            var_name = f'R{i}'
            if var_name in self.var_arrays[var_name]:
                terms = self.var_arrays[var_name]
                phases = terms[:, 1:2]
                freqs = terms[:, 2:3]
                amps = terms[:, 0]
                
                cos_vals = np.cos(phases + freqs * t[None, :])
                R += t_power * (amps @ cos_vals)
        
        # 规范化
        L = L % (2 * np.pi)
        B = B % (2 * np.pi)
        
        return L, B, R


def generate_sample_vsop87_data(output_path: str):
    """
    生成示例 VSOP87 数据文件（用于测试）
    
    注意：这是简化的示例数据，仅用于演示。
    实际使用需要下载完整的 VSOP87 系数表。
    """
    sample_data = """# VSOP87 地球数据示例
# 格式: 变量索引 幂次 项序号 振幅 相位 频率
# 黄经 L0 项（幂次 0）
1 0 1 1.7535935 0.0000000 6283.0758500
1 0 2 0.0334166 4.6692568 6283.0758500
1 0 3 0.0003489 4.6261000 12566.1517000
# 黄经 L1 项（幂次 1）
1 1 1 6283.3196675 0.0000000 0.0000000
1 1 2 0.0020601 2.6782346 6283.0758500
# 黄纬 B0 项
2 0 1 0.0000000 0.0000000 0.0000000
# 距离 R0 项
3 0 1 1.0001394 0.0000000 0.0000000
3 0 2 0.0167069 3.0984635 6283.0758500
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"示例数据已生成: {output_path}")


def demo():
    """演示 VSOP87 地球坐标计算"""
    # 生成示例数据
    data_path = "vsop87_earth_sample.txt"
    generate_sample_vsop87_data(data_path)
    
    # 初始化计算器
    calc = VSOP87Earth(data_path)
    
    # 计算 J2000.0 时刻的坐标
    jd_j2000 = 2451545.0
    L, B, R = calc.compute_coordinates(jd_j2000)
    
    print("\n=== VSOP87 地球坐标计算演示 ===")
    print(f"儒略日 JD = {jd_j2000} (J2000.0)")
    print(f"黄经 L = {np.degrees(L):.6f}°")
    print(f"黄纬 B = {np.degrees(B):.6f}°")
    print(f"距离 R = {R:.6f} AU")
    
    # 批量计算演示
    print("\n=== 批量计算演示 ===")
    jds = np.array([2451545.0, 2451545.0 + 365.25, 2451545.0 + 730.5])
    L_batch, B_batch, R_batch = calc.compute_batch(jds)
    
    for i, jd in enumerate(jds):
        print(f"JD {jd}: L={np.degrees(L_batch[i]):.4f}°, B={np.degrees(B_batch[i]):.4f}°, R={R_batch[i]:.6f} AU")


if __name__ == "__main__":
    demo()
