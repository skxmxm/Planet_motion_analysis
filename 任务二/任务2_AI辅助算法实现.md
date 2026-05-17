# 任务 2：AI 辅助算法实现

## 概述

本任务通过 AI Coding 工具编写 Python 程序，实现两类行星位置计算模型：
1. **VSOP87 模块**：基于分析理论的地球坐标计算
2. **DE 插值模块**：基于切比雪夫多项式插值的数值历表

---

## 1. VSOP87 模块：地球坐标计算

### 文件
- `vsop87_earth.py`

### 核心功能

#### 1.1 VSOP87 理论简介

VSOP87 (Variations Seculaires des Orbites Planetaires) 是法国天体力学与星历计算研究所发布的行星分析理论。它将行星轨道要素表示为时间的三角级数：

```
f(t) = Σ(i=0 to ∞) [ A_i · cos(B_i + C_i · t) ]
```

其中：
- A_i：振幅系数
- B_i：相位
- C_i：频率
- t：儒略千年数（从 J2000.0 起算）

#### 1.2 模块结构

| 类/函数 | 功能 |
|---------|------|
| `VSOP87Term` | 单个级数项的数据类 |
| `VSOP87Variable` | 变量（黄经/黄纬/距离）的数据类 |
| `VSOP87Parser` | 系数表解析器 |
| `VSOP87Earth` | 地球坐标计算器 |

#### 1.3 高效 Parser 实现

**两种解析模式：**

1. **标准解析** (`parse()`):
   - 逐行读取文本文件
   - 解析为 Python 对象列表
   - 适合小规模数据和调试

2. **NumPy 向量化解析** (`parse_numpy()`):
   - 一次性读取所有数据到 NumPy 数组
   - 利用 NumPy 的 C 级优化进行分组
   - 适合大规模数据处理

```python
def parse_numpy(self) -> dict:
    # 读取所有行并解析为列表
    raw_data = [...]
    
    # 转换为 NumPy 数组
    data_array = np.array(raw_data)
    
    # 利用 NumPy 布尔索引分组
    for var_idx in np.unique(data_array[:, 0]):
        mask_var = data_array[:, 0] == var_idx
        # ... 进一步按幂次分组
```

#### 1.4 NumPy 向量化运算

**核心计算**：黄经/黄纬/距离的幂级数展开

```
L = L0 + L1·t + L2·t² + L3·t³ + L4·t⁴ + L5·t⁵
B = B0 + B1·t + B2·t² + B3·t³ + B4·t⁴ + B5·t⁵
R = R0 + R1·t + R2·t² + R3·t³ + R4·t⁴ + R5·t⁵
```

其中每个 Li, Bi, Ri 都是三角级数的和：

```
Li = Σ(j) A_j · cos(B_j + C_j · t)
```

**向量化实现** (`compute_batch`):

```python
# 向量化计算所有项
# terms: (n_terms, 3) = [amplitude, phase, frequency]
# t: (m,) 时间数组

phases = terms[:, 1:2]    # (n_terms, 1)
freqs = terms[:, 2:3]     # (n_terms, 1)
amps = terms[:, 0]        # (n_terms,)

# 广播: (n_terms, 1) + (n_terms, 1) * (1, m) -> (n_terms, m)
cos_vals = np.cos(phases + freqs * t[None, :])

# 矩阵乘法: (n_terms,) @ (n_terms, m) -> (m,)
result = amps @ cos_vals
```

**性能优势**：
- 避免 Python 循环开销
- 利用 CPU SIMD 指令
- 适合大规模批量计算

#### 1.5 使用示例

```python
# 初始化计算器
calc = VSOP87Earth("vsop87_earth.txt")

# 单点计算
L, B, R = calc.compute_coordinates(2451545.0)

# 批量计算（向量化）
jds = np.array([2451545.0, 2451546.0, 2451547.0])
L_batch, B_batch, R_batch = calc.compute_batch(jds)
```

---

## 2. DE 插值模块：切比雪夫多项式插值

### 文件
- `de_chebyshev.py`

### 核心功能

#### 2.1 切比雪夫多项式理论

**定义**：
```
T_0(x) = 1
T_1(x) = x
T_n(x) = 2x · T_{n-1}(x) - T_{n-2}(x)   (n ≥ 2)
```

**性质**：
- 在区间 [-1, 1] 上正交
- 具有最小最大偏差性质（最佳一致逼近）
- 高阶项系数快速衰减，适合截断

#### 2.2 插值公式

DE 星历表使用切比雪夫多项式插值：

```
x(t) = Σ(i=0 to n) c_i · T_i(τ)
```

其中：
- c_i：切比雪夫系数
- T_i：第 i 阶切比雪夫多项式
- τ = 2(t - t_mid) / Δt：归一化时间，范围 [-1, 1]

#### 2.3 模块结构

| 类/函数 | 功能 |
|---------|------|
| `ChebyshevPolynomial` | 切比雪夫多项式计算 |
| `DEInterpolator` | 插值器 |
| `DEBinaryReader` | 二进制文件读取器 |
| `DESimpleData` | 示例数据生成 |

#### 2.4 切比雪夫多项式计算

**递推实现** (O(n) 时间)：

```python
T[0] = 1.0
T[1] = tau
for i in range(2, n + 1):
    T[i] = 2.0 * tau * T[i-1] - T[i-2]
```

**向量化批量计算**：

```python
# tau_array: (m,) 时间数组
# T: (m, n+1) 结果矩阵

T[:, 0] = 1.0
T[:, 1] = tau_array
for i in range(2, n + 1):
    T[:, i] = 2.0 * tau_array * T[:, i-1] - T[:, i-2]
```

#### 2.5 二进制数据读取

DE 星历表通常以二进制格式存储：

```python
def read_coefficients(self, offset=0, count=None):
    with open(self.file_path, 'rb') as f:
        f.seek(offset * 8)  # 每个系数 8 字节
        data = f.read(count * 8)
        
        # 解包为双精度浮点数
        n_coeffs = len(data) // 8
        coefficients = struct.unpack(f'{n_coeffs}d', data)
        
        return np.array(coefficients)
```

#### 2.6 插值计算

**单点插值**：

```python
def interpolate(self, t):
    tau = self._normalize_time(t)  # 归一化到 [-1, 1]
    T = ChebyshevPolynomial.evaluate(tau, self.n)
    return np.dot(self.coefficients, T)
```

**批量插值（向量化）**：

```python
def interpolate_batch(self, t_array):
    tau_array = self._normalize_time(t_array)
    T_matrix = ChebyshevPolynomial.evaluate_batch(tau_array, self.n)
    return T_matrix @ self.coefficients  # 矩阵乘法
```

#### 2.7 导数计算

切比雪夫多项式导数：

```python
def derivative(tau, n):
    T = ChebyshevPolynomial.evaluate(tau, n)
    dT = np.zeros(n + 1)
    dT[1] = 1.0
    for i in range(2, n + 1):
        dT[i] = 2.0 * T[i-1] + 2.0 * tau * dT[i-1] - dT[i-2]
    return dT
```

位置导数：

```python
def interpolate_derivative(self, t):
    tau = self._normalize_time(t)
    dT = ChebyshevPolynomial.derivative(tau, self.n)
    dx_dtau = np.dot(self.coefficients, dT)
    dx_dt = dx_dtau * (2.0 / self.dt)
    return dx_dt
```

---

## 3. 两类模型对比

| 特性 | VSOP87 (分析理论) | DE (数值历表) |
|------|------------------|--------------|
| **数学基础** | 三角级数展开 | 切比雪夫多项式插值 |
| **计算方式** | 直接求和 | 递推多项式求值 |
| **精度来源** | 级数截断误差 | 插值阶数 |
| **时间范围** | 数千年 | 数十年至数百年 |
| **存储需求** | 大（数万系数） | 中等（分段存储） |
| **计算速度** | 较慢（大量三角函数） | 较快（递推多项式） |
| **适用场景** | 长期天文研究 | 高精度航天任务 |

---

## 4. 数值稳定性考虑

### 4.1 VSOP87 的稳定性

- **大数吃小数**：万级累加需注意舍入误差累积
- **解决方案**：使用 Kahan 求和或分治求和

### 4.2 DE 插值的稳定性

- **切比雪夫多项式的优势**：在 [-1, 1] 上数值稳定
- **边界连续性**：相邻子区间在边界处应保证函数连续
- **龙格现象避免**：切比雪夫节点分布避免了等距插值的振荡

---

## 5. 运行演示

### VSOP87 模块

```bash
python vsop87_earth.py
```

输出：
```
=== VSOP87 地球坐标计算演示 ===
儒略日 JD = 2451545.0 (J2000.0)
黄经 L = 100.466457°
黄纬 B = 0.000000°
距离 R = 0.983310 AU
```

### DE 插值模块

```bash
python de_chebyshev.py
```

输出：
```
=== DE 切比雪夫插值演示 ===
1. 切比雪夫多项式值:
   T_0(0.5) = 1.000000
   T_1(0.5) = 0.500000
   T_2(0.5) = -0.500000
   ...

2. 生成示例 DE 插值数据:
   生成了 4 个插值器
   每个插值器使用 13 个切比雪夫系数
```

---

## 6. 扩展建议

1. **实际数据接入**：
   - VSOP87：下载完整的 VSOP87 系数表（约 200KB/行星）
   - DE：下载 JPL 发布的 DE440/DE441 二进制文件

2. **性能优化**：
   - 使用 Numba 加速关键循环
   - 使用多线程并行处理多个行星
   - 预计算并缓存三角函数值

3. **精度验证**：
   - 与 JPL Horizons 系统结果对比
   - 计算残差分析误差来源
