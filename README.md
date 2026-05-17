# Planet Motion Analysis

行星运动数值分析项目 - 基于 VSOP87 分析理论与 DE430 数值历表的对比研究

## 项目简介

本项目是天体物理与科学计算交叉领域的实验项目，通过对比两种主流行星位置描述方法，深入理解数值计算中的误差来源、传播规律及模型选择策略。

### 两种主流方法

| 方法 | 类型 | 特点 |
|------|------|------|
| **VSOP87** | 分析理论 | 三角级数展开，长期外推会累积截断误差 |
| **DE430** | 数值历表 | 切比雪夫多项式插值，精度高但时间范围有限 |

## 项目结构

```
Planet_motion_analysis/
├── 任务一/                          # 任务 1：误差来源的理论映射
│   └── 任务1_误差来源的理论映射.md
├── 任务二/                          # 任务 2：AI 辅助算法实现
│   ├── vsop87_earth.py             # VSOP87 地球坐标计算模块
│   ├── de_chebyshev.py             # DE 切比雪夫插值模块
│   └── 任务2_AI辅助算法实现.md
├── 任务三/                          # 任务 3：误差传播与稳定性实验
│   ├── error_analysis.py           # 误差分析主程序
│   └── 任务3_误差传播与稳定性实验.md
├── planetary_ephemeris_agent.py    # 行星历表智能体
├── 实验报告.tex                     # LaTeX 实验报告
├── requirements.txt                 # Python 依赖
└── README.md                        # 本文件
```

## 核心功能

### 1. VSOP87 模块 (`vsop87_earth.py`)

- **高效 Parser**：支持 NumPy 向量化解析系数表
- **批量计算**：利用矩阵乘法实现万级项数快速求和
- **Kahan 求和**：可选的补偿求和算法减少舍入误差累积

```python
from 任务二.vsop87_earth import VSOP87Earth

calc = VSOP87Earth("vsop87_earth.txt")
L, B, R = calc.compute_coordinates(2451545.0)  # J2000.0 时刻坐标
```

### 2. DE 插值模块 (`de_chebyshev.py`)

- **切比雪夫多项式**：递推计算 + 向量化批量计算
- **高阶插值**：支持任意阶数切比雪夫插值
- **边界检查**：严格的输入时间范围验证
- **连续性验证**：函数值 + 导数双重连续性检查

```python
from 任务二.de_chebyshev import DEInterpolator

interp = DEInterpolator(coeffs, t_start, t_end)
result = interp.interpolate(jd)           # 单点插值
results = interp.interpolate_batch(jds)   # 批量插值
```

### 3. 行星历表智能体 (`planetary_ephemeris_agent.py`)

**自然语言查询 → 自动模型选择 → 误差估计 → 可视化**

```python
from planetary_ephemeris_agent import PlanetaryEphemerisAgent

agent = PlanetaryEphemerisAgent()
result = agent.query("查询 2050 年火星位置，要求精度小于 1e-6")
agent.generate_comparison_plot(result, "mars_2050.png")
```

**智能体特性**：
- 自动解析年份、行星、精度要求
- 智能选择 VSOP87 / DE430 模型
- 给出误差估计和有效数字判定
- 自动生成轨道差异对比图

### 4. 误差分析模块 (`error_analysis.py`)

- **时间演化实验**：计算 1000 年误差增长曲线
- **有效数字判定**：根据相对误差估计精度位数
- **误差增长拟合**：对数线性拟合分析趋势

## 安装与使用

### 环境要求

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

### 安装依赖

```bash
pip install -r requirements.txt
```

### 快速开始

```bash
# 运行 VSOP87 演示
python 任务二/vsop87_earth.py

# 运行 DE 插值演示
python 任务二/de_chebyshev.py

# 运行误差分析实验
python 任务三/error_analysis.py

# 运行智能体演示
python planetary_ephemeris_agent.py
```

## 关键实验结果

### 误差增长趋势

| 时间跨度 | 黄经误差 | 距离相对误差 | 有效数字 |
|---------|---------|------------|---------|
| 0 年 | 23 角秒 | 1.42×10⁻⁴ | ~4.8 |
| 500 年 | 24448 角秒 | 1.40×10⁻⁴ | ~2.2 |
| 1000 年 | - | - | ~1.9 |

### 模型选择建议

| 应用场景 | 推荐模型 | 时间范围 |
|---------|---------|---------|
| 日常天文观测 | VSOP87 | < 100 年 |
| 航天任务规划 | DE430/DE440 | < 50 年 |
| 长期历史研究 | DE 系列 | 任意 |
| 考古天文学 | 数值积分 | 数千年 |

## Prompt Engineering 案例

本项目展示了如何通过 Prompt Engineering 纠正 AI 在复杂数值算法中的逻辑错误：

### 发现的问题

1. **时间范围检查缺失**：切比雪夫插值未验证输入时间是否在有效区间
2. **导数连续性未验证**：DE 星历表要求 C¹ 连续，但初始代码只检查函数值

### 纠正过程

```
初始 Prompt → 基础代码框架
     ↓
纠错 Prompt → 添加边界检查 + 导数连续性验证
     ↓
完善 Prompt → 质量评估 + 错误处理
```

详见 `实验报告.tex` 第 2 章。

## 课程知识关联

本项目涉及科学计算课程的核心知识点：

| 知识点 | 应用 |
|--------|------|
| **误差类型** | 模型误差、截断误差、舍入误差的识别与分析 |
| **有效数字** | 精度判定与误差传播估计 |
| **插值方法** | 切比雪夫多项式插值的实现与优化 |
| **数值稳定性** | 长时间积分的误差累积分析 |
| **向量化运算** | NumPy 加速大规模天文计算 |

## 作者

- 科学计算课程实验项目
- 使用 AI 编程工具辅助开发

## 许可证

MIT License

## 致谢

- VSOP87 理论：法国天体力学与星历计算研究所
- DE430 历表：NASA JPL
- 课程教师与助教的指导
