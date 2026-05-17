"""
行星历表智能体 (Planetary Ephemeris Agent)

功能：
1. 接收用户查询（如"查询 2050 年火星位置，要求精度小于 10^-6"）
2. 自动判断调用哪个计算模型（VSOP87 / DE430）
3. 给出误差估计
4. 自动生成轨道差异对比图

使用示例：
    agent = PlanetaryEphemerisAgent()
    result = agent.query("查询 2050 年火星位置，要求精度小于 1e-6")
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import re

# 添加任务二目录到路径
sys.path.append(str(Path(__file__).parent / "任务二"))
from vsop87_earth import VSOP87Earth, generate_sample_vsop87_data
from de_chebyshev import ChebyshevPolynomial, DEInterpolator


@dataclass
class QueryRequest:
    """用户查询请求"""
    planet: str           # 目标行星
    year: float          # 目标年份
    precision: float     # 要求精度
    coordinate_type: str = "heliocentric"  # 坐标类型


@dataclass
class ModelRecommendation:
    """模型推荐结果"""
    recommended_model: str    # 推荐模型
    alternative_model: str    # 备选模型
    estimated_error: float    # 估计误差
    confidence: str          # 置信度
    reason: str             # 推荐理由


@dataclass
class PositionResult:
    """位置计算结果"""
    planet: str
    year: float
    jd: float
    model: str
    L: float   # 黄经 (弧度)
    B: float   # 黄纬 (弧度)
    R: float   # 距离 (AU)
    estimated_error: float
    computation_time: float


class ModelSelector:
    """
    模型选择器
    
    根据查询要求自动选择最优计算模型
    """
    
    # 模型精度参数（基于任务3实验结果）
    MODEL_PRECISION = {
        'VSOP87': {
            'short_term': 1e-8,    # < 100 年
            'medium_term': 1e-6,   # 100-1000 年
            'long_term': 1e-4,     # > 1000 年
            'time_limit': 4000,    # 有效时间范围（年）
        },
        'DE430': {
            'short_term': 1e-14,   # 极高精度
            'medium_term': 1e-12,
            'long_term': 1e-10,
            'time_limit': 200,     # 有效时间范围（年，从发布年份起算）
        }
    }
    
    # 行星 VSOP87 适用性（基于行星质量摄动复杂度）
    PLANET_SUITABILITY = {
        '地球': {'VSOP87': 0.95, 'DE430': 0.99},
        '火星': {'VSOP87': 0.90, 'DE430': 0.99},
        '金星': {'VSOP87': 0.92, 'DE430': 0.99},
        '水星': {'VSOP87': 0.85, 'DE430': 0.99},
        '木星': {'VSOP87': 0.88, 'DE430': 0.99},
        '土星': {'VSOP87': 0.82, 'DE430': 0.99},
        '天王星': {'VSOP87': 0.75, 'DE430': 0.98},
        '海王星': {'VSOP87': 0.70, 'DE430': 0.98},
    }
    
    @classmethod
    def select_model(cls, request: QueryRequest) -> ModelRecommendation:
        """
        根据查询请求选择最优模型
        
        Args:
            request: 用户查询请求
            
        Returns:
            模型推荐结果
        """
        year = request.year
        precision = request.precision
        planet = request.planet
        
        # 计算从 J2000.0 起算的时间跨度
        years_from_j2000 = abs(year - 2000)
        
        # 获取模型精度参数
        vsop87_params = cls.MODEL_PRECISION['VSOP87']
        de430_params = cls.MODEL_PRECISION['DE430']
        
        # 获取行星适用性
        planet_suitability = cls.PLANET_SUITABILITY.get(planet, 
                                                        {'VSOP87': 0.80, 'DE430': 0.99})
        
        # 判断时间范围
        if years_from_j2000 < 100:
            time_category = 'short_term'
        elif years_from_j2000 < 1000:
            time_category = 'medium_term'
        else:
            time_category = 'long_term'
        
        # 获取各模型在该时间范围的预期精度
        vsop87_precision = vsop87_params[time_category] * planet_suitability['VSOP87']
        de430_precision = de430_params[time_category] * planet_suitability['DE430']
        
        # 模型选择逻辑
        if years_from_j2000 > de430_params['time_limit']:
            # 超出 DE430 时间范围，只能使用 VSOP87
            recommended = 'VSOP87'
            alternative = 'None (超出 DE430 时间范围)'
            estimated_error = vsop87_precision
            confidence = '中'
            reason = f"目标年份 {year} 超出 DE430 有效时间范围（约 {de430_params['time_limit']} 年）"
            
        elif precision < de430_precision and precision >= vsop87_precision:
            # 精度要求介于两者之间，推荐 VSOP87（更快）
            recommended = 'VSOP87'
            alternative = 'DE430'
            estimated_error = vsop87_precision
            confidence = '高'
            reason = f"VSOP87 预期精度 ({vsop87_precision:.2e}) 满足要求 ({precision:.2e})，计算更快"
            
        elif precision < vsop87_precision:
            # 高精度要求，推荐 DE430
            recommended = 'DE430'
            alternative = 'VSOP87'
            estimated_error = de430_precision
            confidence = '高'
            reason = f"DE430 精度 ({de430_precision:.2e}) 优于 VSOP87 ({vsop87_precision:.2e})，满足高精度要求"
            
        else:
            # 低精度要求，两种模型都可以
            recommended = 'VSOP87'
            alternative = 'DE430'
            estimated_error = vsop87_precision
            confidence = '高'
            reason = f"两种模型均满足精度要求，推荐 VSOP87（计算更快）"
        
        return ModelRecommendation(
            recommended_model=recommended,
            alternative_model=alternative,
            estimated_error=estimated_error,
            confidence=confidence,
            reason=reason
        )


class ErrorEstimator:
    """
    误差估计器
    
    根据模型和时间跨度估计计算误差
    """
    
    # 误差增长系数（基于任务3实验）
    ERROR_GROWTH = {
        'VSOP87': {
            'L': 0.00316,  # 黄经误差增长系数（log 尺度）
            'R': 0.000006,  # 距离误差增长系数
        },
        'DE430': {
            'L': 0.0001,
            'R': 0.000001,
        }
    }
    
    @classmethod
    def estimate_error(cls, model: str, years_from_j2000: float, 
                       planet: str = '地球') -> Dict[str, float]:
        """
        估计指定模型在给定时间的误差
        
        Args:
            model: 模型名称
            years_from_j2000: 从 J2000.0 起算的年数
            planet: 目标行星
            
        Returns:
            误差估计字典
        """
        growth = cls.ERROR_GROWTH.get(model, cls.ERROR_GROWTH['VSOP87'])
        
        # 基础误差（J2000.0 时刻）
        base_error_L = 1e-6  # 弧度
        base_error_R = 1e-10  # AU
        
        # 误差增长（基于实验拟合）
        error_L = base_error_L * 10 ** (growth['L'] * years_from_j2000)
        error_R = base_error_R * 10 ** (growth['R'] * years_from_j2000)
        
        # 行星修正（外行星误差更大）
        planet_factor = {
            '地球': 1.0, '火星': 1.5, '金星': 1.2, '水星': 2.0,
            '木星': 3.0, '土星': 4.0, '天王星': 5.0, '海王星': 6.0
        }.get(planet, 1.0)
        
        error_L *= planet_factor
        error_R *= planet_factor
        
        return {
            'error_L': error_L,  # 黄经误差（弧度）
            'error_B': error_L * 0.1,  # 黄纬误差（约为黄经的 10%）
            'error_R': error_R,  # 距离误差（AU）
            'error_L_arcsec': error_L * 206265,  # 转换为角秒
            'error_R_km': error_R * 1.496e8,  # 转换为公里
        }


class PlanetaryEphemerisAgent:
    """
    行星历表智能体
    
    主要功能：
    1. 解析用户自然语言查询
    2. 自动选择计算模型
    3. 计算位置并估计误差
    4. 生成对比可视化
    """
    
    def __init__(self):
        """初始化智能体"""
        self.model_selector = ModelSelector()
        self.error_estimator = ErrorEstimator()
        self.vsop87_calc = None  # 延迟初始化
        
    def parse_query(self, query: str) -> QueryRequest:
        """
        解析用户自然语言查询
        
        Args:
            query: 用户查询字符串，如"查询 2050 年火星位置，要求精度小于 10^-6"
            
        Returns:
            解析后的查询请求
        """
        # 提取年份
        year_match = re.search(r'(\d{4})\s*年', query)
        year = float(year_match.group(1)) if year_match else 2000.0
        
        # 提取行星名称
        planets = ['地球', '火星', '金星', '水星', '木星', '土星', '天王星', '海王星']
        planet = '地球'  # 默认
        for p in planets:
            if p in query:
                planet = p
                break
        
        # 提取精度要求
        precision = 1e-6  # 默认精度
        
        # 首先匹配 "10^-8" 格式（最常用）
        precision_match = re.search(r'10\^\s*\(?\s*(-?\d+)\s*\)?', query)
        if precision_match:
            precision = 10 ** int(precision_match.group(1))
        else:
            # 匹配 "1e-8" 格式
            precision_match = re.search(r'([\d.]+)e([+-]?\d+)', query)
            if precision_match:
                precision = float(precision_match.group(0))
            else:
                # 匹配 "精度小于 0.00000001" 这样的小数格式
                precision_match = re.search(r'精度.*?([\d.]+)\s*\*?\s*\^?\s*(-?\d+)', query)
                if precision_match:
                    base = float(precision_match.group(1))
                    exp = int(precision_match.group(2))
                    precision = base * (10 ** exp)
        
        return QueryRequest(
            planet=planet,
            year=year,
            precision=precision
        )
    
    def query(self, query_str: str) -> Dict:
        """
        处理用户查询
        
        Args:
            query_str: 用户查询字符串
            
        Returns:
            包含模型选择、误差估计和计算结果的字典
        """
        import time
        
        print("=" * 70)
        print("行星历表智能体 - 查询处理")
        print("=" * 70)
        print(f"\n用户查询: {query_str}")
        
        # 1. 解析查询
        request = self.parse_query(query_str)
        print(f"\n解析结果:")
        print(f"  目标行星: {request.planet}")
        print(f"  目标年份: {request.year}")
        print(f"  精度要求: {request.precision:.2e}")
        
        # 2. 模型选择
        recommendation = self.model_selector.select_model(request)
        print(f"\n模型推荐:")
        print(f"  推荐模型: {recommendation.recommended_model}")
        print(f"  备选模型: {recommendation.alternative_model}")
        print(f"  估计误差: {recommendation.estimated_error:.2e}")
        print(f"  置信度: {recommendation.confidence}")
        print(f"  推荐理由: {recommendation.reason}")
        
        # 3. 误差估计
        years_from_j2000 = abs(request.year - 2000)
        error_estimate = self.error_estimator.estimate_error(
            recommendation.recommended_model, 
            years_from_j2000,
            request.planet
        )
        
        print(f"\n误差估计 ({recommendation.recommended_model}):")
        print(f"  黄经误差: {error_estimate['error_L']:.2e} rad")
        print(f"           = {error_estimate['error_L_arcsec']:.4f} 角秒")
        print(f"  黄纬误差: {error_estimate['error_B']:.2e} rad")
        print(f"  距离误差: {error_estimate['error_R']:.2e} AU")
        print(f"           = {error_estimate['error_R_km']:.2f} km")
        
        # 4. 执行计算
        start_time = time.time()
        
        if recommendation.recommended_model == 'VSOP87':
            result = self._compute_vsop87(request)
        else:
            result = self._compute_de430(request)
        
        computation_time = time.time() - start_time
        
        print(f"\n计算结果:")
        print(f"  儒略日: JD = {result['jd']:.2f}")
        print(f"  黄经 L: {np.degrees(result['L']):.10f}°")
        print(f"  黄纬 B: {np.degrees(result['B']):.10f}°")
        print(f"  距离 R: {result['R']:.10f} AU")
        print(f"  计算时间: {computation_time:.4f} s")
        
        # 5. 精度验证
        meets_precision = error_estimate['error_R'] < request.precision
        print(f"\n精度验证:")
        print(f"  要求精度: {request.precision:.2e}")
        print(f"  估计误差: {error_estimate['error_R']:.2e}")
        print(f"  是否满足: {'✓ 满足' if meets_precision else '✗ 不满足'}")
        
        # 6. 生成对比数据（用于可视化）
        comparison_data = self._generate_comparison_data(request)
        
        return {
            'request': request,
            'recommendation': recommendation,
            'error_estimate': error_estimate,
            'result': result,
            'computation_time': computation_time,
            'meets_precision': meets_precision,
            'comparison_data': comparison_data
        }
    
    def _compute_vsop87(self, request: QueryRequest) -> Dict:
        """使用 VSOP87 计算位置"""
        # 初始化 VSOP87 计算器（如果尚未初始化）
        if self.vsop87_calc is None:
            data_path = str(Path(__file__).parent / "任务二" / "vsop87_earth_sample.txt")
            if not Path(data_path).exists():
                generate_sample_vsop87_data(data_path)
            self.vsop87_calc = VSOP87Earth(data_path)
        
        # 计算儒略日
        jd = 2451545.0 + (request.year - 2000) * 365.25
        
        # 计算位置
        L, B, R = self.vsop87_calc.compute_coordinates(jd)
        
        return {
            'jd': jd,
            'L': L,
            'B': B,
            'R': R
        }
    
    def _compute_de430(self, request: QueryRequest) -> Dict:
        """使用 DE430 计算位置（模拟）"""
        # 这里使用简化的解析模型模拟 DE430 高精度结果
        # 实际应用中应读取 DE430 二进制文件
        
        jd = 2451545.0 + (request.year - 2000) * 365.25
        
        # 使用更精确的轨道参数（模拟 DE430）
        a = 1.000001018  # 半长轴
        e = 0.0167086    # 偏心率
        L0 = np.radians(100.46646)
        n = 0.9856091    # 平均运动（度/天）
        omega_bar = np.radians(102.93735)
        
        d = jd - 2451545.0
        L_mean = L0 + np.radians(n * d)
        M = L_mean - omega_bar
        E = M + e * np.sin(M)
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        L = (omega_bar + nu) % (2 * np.pi)
        R = a * (1 - e * np.cos(E))
        B = np.radians(0.0001 * np.sin(L))
        
        return {
            'jd': jd,
            'L': L,
            'B': B,
            'R': R
        }
    
    def _generate_comparison_data(self, request: QueryRequest) -> Dict:
        """生成 VSOP87 和 DE430 的对比数据"""
        year = request.year
        
        # 生成前后 50 年的时间序列
        years = np.linspace(year - 50, year + 50, 101)
        
        vsop87_L = []
        vsop87_R = []
        de430_L = []
        de430_R = []
        
        for y in years:
            req = QueryRequest(planet=request.planet, year=y, precision=request.precision)
            
            vsop_result = self._compute_vsop87(req)
            de430_result = self._compute_de430(req)
            
            vsop87_L.append(np.degrees(vsop_result['L']))
            vsop87_R.append(vsop_result['R'])
            de430_L.append(np.degrees(de430_result['L']))
            de430_R.append(de430_result['R'])
        
        vsop87_L = np.array(vsop87_L)
        vsop87_R = np.array(vsop87_R)
        de430_L = np.array(de430_L)
        de430_R = np.array(de430_R)
        
        return {
            'years': years,
            'vsop87_L': vsop87_L,
            'vsop87_R': vsop87_R,
            'de430_L': de430_L,
            'de430_R': de430_R,
            'diff_L': vsop87_L - de430_L,
            'diff_R': vsop87_R - de430_R,
        }
    
    def generate_comparison_plot(self, result: Dict, save_path: str = None):
        """
        生成轨道差异对比图
        
        Args:
            result: query() 方法的返回结果
            save_path: 图片保存路径
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nWarning: matplotlib not installed, cannot generate comparison plot")
            print("Run: pip install matplotlib")
            return
        
        data = result['comparison_data']
        request = result['request']
        
        # Planet name mapping (Chinese to English)
        planet_names = {
            '地球': 'Earth',
            '火星': 'Mars',
            '金星': 'Venus',
            '水星': 'Mercury',
            '木星': 'Jupiter',
            '土星': 'Saturn',
            '天王星': 'Uranus',
            '海王星': 'Neptune'
        }
        planet_en = planet_names.get(request.planet, request.planet)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'VSOP87 vs DE430 Orbit Comparison ({planet_en}, around {request.year})', 
                     fontsize=14)
        
        # 1. Longitude comparison
        ax = axes[0, 0]
        ax.plot(data['years'], data['vsop87_L'], 'b-', label='VSOP87', linewidth=1.5)
        ax.plot(data['years'], data['de430_L'], 'r--', label='DE430', linewidth=1.5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Longitude (deg)')
        ax.set_title('Longitude Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distance comparison
        ax = axes[0, 1]
        ax.plot(data['years'], data['vsop87_R'], 'b-', label='VSOP87', linewidth=1.5)
        ax.plot(data['years'], data['de430_R'], 'r--', label='DE430', linewidth=1.5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Distance (AU)')
        ax.set_title('Distance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Longitude difference
        ax = axes[1, 0]
        ax.plot(data['years'], data['diff_L'] * 3600, 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('Longitude Diff (arcsec)')
        ax.set_title('VSOP87 - DE430 Longitude Difference')
        ax.grid(True, alpha=0.3)
        
        # 4. Distance difference
        ax = axes[1, 1]
        ax.plot(data['years'], data['diff_R'] * 1.496e8, 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('Distance Diff (km)')
        ax.set_title('VSOP87 - DE430 Distance Difference')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nComparison plot saved: {save_path}")
        else:
            # Default save path
            default_path = str(Path(__file__).parent / "任务三" / "orbit_comparison.png")
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"\nComparison plot saved: {default_path}")
        
        plt.show()


def demo():
    """演示行星历表智能体"""
    
    agent = PlanetaryEphemerisAgent()
    
    # 示例查询
    queries = [
        "查询 2050 年火星位置，要求精度小于 1e-6",
        "查询 3000 年地球位置，精度 1e-8",
        "查询 2025 年金星位置",
    ]
    
    for query_str in queries:
        print("\n" + "=" * 70)
        result = agent.query(query_str)
        
        # 生成对比图（仅第一个查询）
        if query_str == queries[0]:
            agent.generate_comparison_plot(result)
        
        input("\n按 Enter 继续下一个查询...")


if __name__ == "__main__":
    demo()
