"""
任务 3：误差传播与稳定性实验

使用并对比 VSOP87 与 JPL DE430 计算得到的地球位置：
1. 时间演化实验：计算从 J2000.0 历元开始，往前后推 500 年，观察两模型之间的绝对误差随时间的增长曲线
2. 精度判定：根据有效数字定义，判断在计算 1000 年后的位置时，VSOP87 还能保持几位有效数字

注意：由于我们没有真实的 DE430 数据，本实验使用以下方法模拟：
- VSOP87：使用任务2中的简化模型
- DE430 参考值：使用更高精度的数值积分或理论值作为"真实值"
- 为演示目的，我们使用一个简化的参考模型来模拟 DE430 的高精度结果
"""

import numpy as np
import sys
from pathlib import Path

# 添加任务二目录到路径，导入 VSOP87 模块
sys.path.append(str(Path(__file__).parent.parent / "任务二"))
from vsop87_earth import VSOP87Earth, generate_sample_vsop87_data


class ReferenceEphemeris:
    """
    参考历表（模拟 DE430 高精度结果）
    
    使用更精确的解析公式模拟"真实"地球位置
    这个模型比 VSOP87 简化模型更精确
    """
    
    def __init__(self):
        """初始化参考历表"""
        # 地球轨道参数（更精确的值）
        self.a = 1.000001018  # 半长轴 (AU)
        self.e = 0.0167086    # 偏心率
        self.i = np.radians(0.000015)  # 轨道倾角
        self.L0 = np.radians(100.46646)  # 平黄经 J2000.0
        self.omega_bar = np.radians(102.93735)  # 近日点黄经
        self.n = 0.9856091    # 平均运动 (度/天)
        
    def compute_position(self, jd: float) -> tuple:
        """
        计算地球日心黄道坐标（简化但相对精确的模型）
        
        Args:
            jd: 儒略日
            
        Returns:
            (L, B, R) 元组，单位为弧度和 AU
        """
        # 从 J2000.0 起算的天数
        d = jd - 2451545.0
        
        # 平黄经
        L_mean = self.L0 + np.radians(self.n * d)
        
        # 简化计算：考虑偏心率和近日点进动
        # 使用开普勒方程的近似解
        M = L_mean - self.omega_bar  # 平近点角
        
        # 偏近点角的近似（一阶近似）
        E = M + self.e * np.sin(M)
        
        # 真近点角
        nu = 2 * np.arctan(np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(E / 2))
        
        # 黄经
        L = self.omega_bar + nu
        
        # 距离（椭圆轨道）
        R = self.a * (1 - self.e * np.cos(E))
        
        # 黄纬（非常小的值）
        B = np.radians(0.0001 * np.sin(L))
        
        # 规范化
        L = L % (2 * np.pi)
        B = B % (2 * np.pi)
        
        return L, B, R


class ErrorAnalysis:
    """
    误差分析器
    
    对比 VSOP87 和参考历表（模拟 DE430）的结果
    """
    
    def __init__(self, vsop87_data_path: str):
        """
        初始化误差分析器
        
        Args:
            vsop87_data_path: VSOP87 数据文件路径
        """
        self.vsop87 = VSOP87Earth(vsop87_data_path)
        self.reference = ReferenceEphemeris()
    
    def compute_error(self, jd: float) -> dict:
        """
        计算单个时间点的误差
        
        Args:
            jd: 儒略日
            
        Returns:
            误差字典，包含绝对误差和相对误差
        """
        # VSOP87 计算结果
        L_v, B_v, R_v = self.vsop87.compute_coordinates(jd)
        
        # 参考值（模拟 DE430）
        L_r, B_r, R_r = self.reference.compute_position(jd)
        
        # 计算误差
        error_L = abs(L_v - L_r)  # 黄经绝对误差（弧度）
        error_B = abs(B_v - B_r)  # 黄纬绝对误差（弧度）
        error_R = abs(R_v - R_r)  # 距离绝对误差（AU）
        
        # 相对误差
        rel_error_L = error_L / abs(L_r) if L_r != 0 else 0
        rel_error_R = error_R / abs(R_r) if R_r != 0 else 0
        
        return {
            'jd': jd,
            'L_v': L_v, 'B_v': B_v, 'R_v': R_v,
            'L_r': L_r, 'B_r': B_r, 'R_r': R_r,
            'error_L': error_L,
            'error_B': error_B,
            'error_R': error_R,
            'rel_error_L': rel_error_L,
            'rel_error_R': rel_error_R
        }
    
    def time_evolution_experiment(self, years: int = 500, step: int = 10) -> dict:
        """
        时间演化实验：计算误差随时间的增长曲线
        
        Args:
            years: 往前/往后推的年数
            step: 计算步长（年）
            
        Returns:
            包含时间序列和误差序列的字典
        """
        # 生成时间序列（从 J2000.0 往前和往后）
        time_years = np.arange(-years, years + step, step)
        
        # 转换为儒略日
        # 1 年 ≈ 365.25 天
        jds = 2451545.0 + time_years * 365.25
        
        # 计算每个时间点的误差
        errors = []
        for jd in jds:
            error = self.compute_error(jd)
            errors.append(error)
        
        # 提取误差序列
        error_L = np.array([e['error_L'] for e in errors])
        error_B = np.array([e['error_B'] for e in errors])
        error_R = np.array([e['error_R'] for e in errors])
        rel_error_L = np.array([e['rel_error_L'] for e in errors])
        rel_error_R = np.array([e['rel_error_R'] for e in errors])
        
        return {
            'time_years': time_years,
            'jds': jds,
            'error_L': error_L,  # 弧度
            'error_B': error_B,
            'error_R': error_R,  # AU
            'rel_error_L': rel_error_L,
            'rel_error_R': rel_error_R
        }
    
    def analyze_effective_digits(self, years: int = 1000) -> dict:
        """
        精度判定：根据有效数字定义，判断 VSOP87 在指定年份后还能保持几位有效数字
        
        有效数字定义：
        如果近似值 x* 的误差不超过某一位的半个单位，且该位到 x* 的第一位非零数字共有 n 位，
        则称 x* 有 n 位有效数字。
        
        相对误差与有效数字的关系：
        |x - x*| / |x| ≤ 1/(2*a1) * 10^(-(n-1))
        
        其中 a1 是第一位有效数字。
        
        Args:
            years: 计算的年数
            
        Returns:
            包含有效数字分析结果的字典
        """
        jd = 2451545.0 + years * 365.25
        error = self.compute_error(jd)
        
        # 距离的有效数字分析
        R_true = error['R_r']
        R_approx = error['R_v']
        abs_error_R = error['error_R']
        rel_error_R = error['rel_error_R']
        
        # 根据相对误差估计有效数字
        # 简化估计：n ≈ -log10(rel_error) + 1
        if rel_error_R > 0:
            n_digits_R = -np.log10(rel_error_R) + 1
        else:
            n_digits_R = np.inf
        
        # 黄经的有效数字分析
        L_true = error['L_r']
        L_approx = error['L_v']
        abs_error_L = error['error_L']
        rel_error_L = error['rel_error_L']
        
        if rel_error_L > 0:
            n_digits_L = -np.log10(rel_error_L) + 1
        else:
            n_digits_L = np.inf
        
        return {
            'years': years,
            'jd': jd,
            'R_true': R_true,
            'R_approx': R_approx,
            'abs_error_R': abs_error_R,
            'rel_error_R': rel_error_R,
            'n_digits_R': n_digits_R,
            'L_true': L_true,
            'L_approx': L_approx,
            'abs_error_L': abs_error_L,
            'rel_error_L': rel_error_L,
            'n_digits_L': n_digits_L
        }


def print_results(results: dict):
    """打印时间演化实验结果"""
    print("\n" + "="*70)
    print("时间演化实验结果（误差随时间变化）")
    print("="*70)
    
    print(f"\n{'年份':>8} {'黄经误差(角秒)':>15} {'距离误差(AU)':>15} {'距离相对误差':>15}")
    print("-"*60)
    
    for i in range(len(results['time_years'])):
        year = results['time_years'][i]
        # 将弧度转换为角秒（1 弧度 = 206265 角秒）
        error_L_arcsec = results['error_L'][i] * 206265
        error_R = results['error_R'][i]
        rel_error_R = results['rel_error_R'][i]
        
        print(f"{year:>8} {error_L_arcsec:>15.6f} {error_R:>15.8f} {rel_error_R:>15.8f}")


def print_effective_digits(analysis: dict):
    """打印有效数字分析结果"""
    print("\n" + "="*70)
    print(f"精度判定：{analysis['years']} 年后的有效数字分析")
    print("="*70)
    
    print(f"\n儒略日: JD = {analysis['jd']:.2f}")
    
    print("\n【距离 R 的分析】")
    print(f"  参考值 (DE430):     R = {analysis['R_true']:.10f} AU")
    print(f"  VSOP87 计算值:      R = {analysis['R_approx']:.10f} AU")
    print(f"  绝对误差:           |ΔR| = {analysis['abs_error_R']:.10f} AU")
    print(f"  相对误差:           |ΔR|/R = {analysis['rel_error_R']:.10f}")
    print(f"  估计有效数字位数:   n ≈ {analysis['n_digits_R']:.2f} 位")
    
    print("\n【黄经 L 的分析】")
    print(f"  参考值 (DE430):     L = {np.degrees(analysis['L_true']):.10f}°")
    print(f"  VSOP87 计算值:      L = {np.degrees(analysis['L_approx']):.10f}°")
    print(f"  绝对误差:           |ΔL| = {np.degrees(analysis['abs_error_L']):.10f}°")
    print(f"  相对误差:           |ΔL|/L = {analysis['rel_error_L']:.10f}")
    print(f"  估计有效数字位数:   n ≈ {analysis['n_digits_L']:.2f} 位")
    
    # 有效数字判定
    print("\n【有效数字判定结论】")
    n_digits = min(analysis['n_digits_R'], analysis['n_digits_L'])
    if n_digits >= 5:
        print(f"  ✓ VSOP87 在 {analysis['years']} 年后仍保持约 {int(n_digits)} 位有效数字，精度良好")
    elif n_digits >= 3:
        print(f"  △ VSOP87 在 {analysis['years']} 年后约保持 {int(n_digits)} 位有效数字，精度一般")
    else:
        print(f"  ✗ VSOP87 在 {analysis['years']} 年后仅保持约 {int(n_digits)} 位有效数字，精度不足")


def demo():
    """演示误差传播与稳定性实验"""
    
    print("="*70)
    print("任务 3：误差传播与稳定性实验")
    print("="*70)
    
    # 准备 VSOP87 数据
    data_path = "vsop87_earth_sample.txt"
    generate_sample_vsop87_data(data_path)
    
    # 初始化误差分析器
    analyzer = ErrorAnalysis(data_path)
    
    # ========== 实验 1：时间演化实验 ==========
    print("\n" + "="*70)
    print("实验 1：时间演化实验（前后 500 年）")
    print("="*70)
    
    results = analyzer.time_evolution_experiment(years=500, step=50)
    print_results(results)
    
    # 误差增长趋势分析
    print("\n【误差增长趋势分析】")
    
    # 拟合误差增长曲线（假设误差随时间线性或指数增长）
    time_abs = np.abs(results['time_years'])
    
    # 距离误差增长
    log_error_R = np.log10(results['error_R'] + 1e-15)  # 避免 log(0)
    coeffs_R = np.polyfit(time_abs, log_error_R, 1)
    
    print(f"  距离误差增长趋势: log(ε_R) ≈ {coeffs_R[0]:.6f} * |t| + {coeffs_R[1]:.6f}")
    if coeffs_R[0] > 0.01:
        print(f"  → 误差呈指数增长趋势，每 {1/coeffs_R[0]:.1f} 年误差增长约 10 倍")
    else:
        print(f"  → 误差增长较缓慢")
    
    # 黄经误差增长（转换为角秒）
    error_L_arcsec = results['error_L'] * 206265
    log_error_L = np.log10(error_L_arcsec + 1e-15)
    coeffs_L = np.polyfit(time_abs, log_error_L, 1)
    
    print(f"  黄经误差增长趋势: log(ε_L) ≈ {coeffs_L[0]:.6f} * |t| + {coeffs_L[1]:.6f}")
    if coeffs_L[0] > 0.01:
        print(f"  → 误差呈指数增长趋势，每 {1/coeffs_L[0]:.1f} 年误差增长约 10 倍")
    else:
        print(f"  → 误差增长较缓慢")
    
    # ========== 实验 2：精度判定 ==========
    print("\n" + "="*70)
    print("实验 2：精度判定（1000 年后）")
    print("="*70)
    
    analysis = analyzer.analyze_effective_digits(years=1000)
    print_effective_digits(analysis)
    
    # 不同年份的精度对比
    print("\n" + "="*70)
    print("不同年份的精度对比")
    print("="*70)
    print(f"\n{'年份':>8} {'距离有效数字':>15} {'黄经有效数字':>15}")
    print("-"*45)
    
    for years in [100, 500, 1000, 2000, 5000]:
        analysis_year = analyzer.analyze_effective_digits(years=years)
        print(f"{years:>8} {analysis_year['n_digits_R']:>15.2f} {analysis_year['n_digits_L']:>15.2f}")
    
    print("\n" + "="*70)
    print("实验结论")
    print("="*70)
    print("""
1. 时间演化实验：
   - VSOP87 与参考历表（DE430）的误差随时间增长
   - 误差增长速率可用于评估 VSOP87 的适用范围
   - 对于高精度要求（如航天任务），需要定期更新星历表

2. 有效数字判定：
   - 有效数字位数随时间推移逐渐减少
   - 在 1000 年时间尺度上，VSOP87 仍能保持一定精度
   - 对于更长时间尺度（数千年），建议使用数值历表（DE 系列）

3. 数值稳定性启示：
   - 截断误差（VSOP87 级数截断）会随时间累积
   - 模型误差（简化假设）是系统性的，不会随计算过程放大
   - 在实际应用中，需要根据精度要求选择合适的时间范围
    """)


if __name__ == "__main__":
    demo()
