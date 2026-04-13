# Mode 7 夜间点云噪点优化实施总结

## 实施日期
2026-04-08

## 问题描述
用户在 Mode 7 (DSG Night) 的点云输出中观察到红圈标注的孤立噪点：
- 左下角和中间点云视图存在黄色/棕色噪点
- 噪点主要出现在草坪区域
- 已有的 speckle 过滤（SpeckleWindowSize=100, SpeckleRange=8）不足以完全去除

## 实施的优化策略

### 策略1：邻域一致性检查 ✅ 已实施

**位置**：`src/stereo_multi_match.cpp` line 859-891

**实现内容**：
在密集采样循环中添加 3x3 邻域一致性检查，过滤孤立噪点。

**核心逻辑**：
```cpp
// 邻域一致性检查：统计3x3窗口内的有效邻居数，过滤孤立噪点
int valid_neighbors = 0;
for (int dy = -1; dy <= 1; dy++)
{
    for (int dx = -1; dx <= 1; dx++)
    {
        if (dy == 0 && dx == 0)
            continue;
        int ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < safe_rows && nx >= 0 && nx < safe_cols)
        {
            uint8_t neighbor_label = lab.at<uchar>(ny, nx);
            float neighbor_depth = depth.at<float>(ny, nx);
            // 邻居有效条件：同类标签（障碍物）且深度相近
            if (neighbor_label >= 100 && neighbor_depth > 0 &&
                neighbor_depth < 6.0f &&
                std::abs(neighbor_depth - d) < 0.3f)
            {
                valid_neighbors++;
            }
        }
    }
}
// 要求至少2个有效邻居才添加点，过滤孤立噪点
if (valid_neighbors < 2)
{
    continue;
}
```

**关键参数**：
- 邻域窗口：3x3（8个邻居）
- 最小有效邻居数：2
- 深度容忍度：0.3m

**优点**：
- 计算成本极低：仅检查8个邻居像素
- 直接在采样阶段过滤，避免生成无效点云
- 对孤立噪点非常有效

**预期效果**：过滤 80-90% 的孤立噪点

### 策略2：增强 RadiusOutlierRemoval 参数 ✅ 已实施

**位置**：`src/stereo_multi_match.cpp` line 1009-1010

**修改内容**：
```cpp
// 修改前
ror.setRadiusSearch(0.08);      // 8cm搜索半径
ror.setMinNeighborsInRadius(6); // 至少6个邻居

// 修改后
ror.setRadiusSearch(0.10);      // 从 8cm 增加到 10cm，更严格过滤夜间噪点
ror.setMinNeighborsInRadius(8); // 从 6 增加到 8，要求更多邻居
```

**参数变化**：
- 搜索半径：0.08m → 0.10m（增加 25%）
- 最小邻居数：6 → 8（增加 33%）

**优点**：
- 修改简单，仅两个参数
- 在已有的过滤流程中增强效果
- 对整体性能影响很小（在降采样后的点云上操作）

**预期效果**：进一步过滤剩余的稀疏噪点

## 性能影响评估

### 策略1 - 邻域一致性检查
- **额外计算**：每个障碍物像素检查 8 个邻居
- **预估性能影响**：< 2ms/帧
- **原因**：简单的数组访问和比较操作，无复杂计算

### 策略2 - RadiusOutlierRemoval 增强
- **额外计算**：搜索半径增大 25%，邻居数要求增加 33%
- **预估性能影响**：< 1ms/帧
- **原因**：在降采样后的点云上操作，点数已大幅减少

### 总体性能影响
- **预估总影响**：< 3ms/帧
- **可接受性**：在可接受范围内（目标 < 5ms）

## 验证方法

### 测试命令
```bash
# 编译
./build.sh

# 运行 Mode 7 处理夜间图像
./build/offline_perception_debug_432_sob --mode 7 --input <night_image_path>
```

### 评估指标
1. **噪点数量**：红圈区域的孤立黄色/棕色点是否明显减少
2. **有效点云**：障碍物点云是否保持完整
3. **处理性能**：帧处理时间增加是否 < 5ms
4. **点云密度**：整体点云数量减少是否 < 10%

### 预期结果
- 策略1：过滤 80-90% 的孤立噪点
- 策略2：进一步过滤剩余噪点
- 总体：噪点显著减少，有效点云保持完整

## 参数调优建议

如果效果不理想，可以调整以下参数：

### 策略1 - 邻域一致性
**位置**：`src/stereo_multi_match.cpp` line 887

**可调参数**：
```cpp
// 当前：要求至少2个有效邻居
if (valid_neighbors < 2)

// 更严格（过滤更多噪点，可能误删有效点）
if (valid_neighbors < 3)

// 更宽松（保留更多点，可能保留噪点）
if (valid_neighbors < 1)
```

**深度容忍度**（line 882）：
```cpp
// 当前：0.3m
std::abs(neighbor_depth - d) < 0.3f

// 更严格
std::abs(neighbor_depth - d) < 0.2f

// 更宽松
std::abs(neighbor_depth - d) < 0.5f
```

### 策略2 - RadiusOutlierRemoval
**位置**：`src/stereo_multi_match.cpp` line 1009-1010

**可调参数**：
```cpp
// 当前配置
ror.setRadiusSearch(0.10);
ror.setMinNeighborsInRadius(8);

// 更严格（过滤更多噪点）
ror.setRadiusSearch(0.12);
ror.setMinNeighborsInRadius(10);

// 更宽松（保留更多点）
ror.setRadiusSearch(0.08);
ror.setMinNeighborsInRadius(6);
```

## 备选策略（如果效果不足）

### 策略3：草坪区域深度一致性检查

**位置**：`src/stereo_multi_match.cpp` line 783 之后

**实现**：在粗采样循环中，针对 label 1-3 添加深度中值检查

**代码**：
```cpp
// 草坪区域深度一致性检查
if (pc_rgbl.label >= 1 && pc_rgbl.label <= 3) {
    // 检查周围5x5窗口的深度中值
    std::vector<float> neighbor_depths;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < safe_rows && nx >= 0 && nx < safe_cols) {
                float nd = depth.at<float>(ny, nx);
                if (nd > 0 && nd < 6.0f) {
                    neighbor_depths.push_back(nd);
                }
            }
        }
    }
    if (!neighbor_depths.empty()) {
        std::nth_element(neighbor_depths.begin(), 
                        neighbor_depths.begin() + neighbor_depths.size()/2,
                        neighbor_depths.end());
        float median_depth = neighbor_depths[neighbor_depths.size()/2];
        // 如果深度偏离中值超过0.5m，跳过
        if (std::abs(d - median_depth) > 0.5f) {
            continue;
        }
    }
}
```

**性能影响**：< 3ms/帧（5x5 窗口计算）

**建议**：仅在策略1+2效果不足时启用

## 代码修改位置总结

| 策略 | 文件 | 行号 | 修改内容 |
|------|------|------|----------|
| 策略1 | stereo_multi_match.cpp | 859-891 | 添加邻域一致性检查 |
| 策略2 | stereo_multi_match.cpp | 1009-1010 | 增强 RadiusOutlierRemoval 参数 |

## 编译状态
✅ 编译成功
- 所有目标构建完成
- 无编译错误
- 可执行文件：`./build/offline_perception_debug_432_sob`

## 下一步
1. 运行 Mode 7 处理夜间图像
2. 观察红圈区域的噪点是否显著减少
3. 检查障碍物点云是否保持完整
4. 测量处理时间增加量
5. 根据效果决定是否需要调整参数或启用策略3
