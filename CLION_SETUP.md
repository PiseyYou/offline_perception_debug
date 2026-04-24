# CLion 运行配置说明

## 问题
从CLion直接运行程序时出现以下错误：
```
error while loading shared libraries: libhbdk_sim_x86.so: cannot open shared object file
```
或者：
```
[E][DNN][packed_model.cpp:129][Model] Can not open models/cdt_20251125_640x384.bin, file not exists or no read permission.
```

## 解决方案

### 方法1：使用包装脚本（最简单）

直接运行包装脚本，它会自动设置所有必要的环境变量：
```bash
./offline_perception_debug_384_sob.sh
```

或从任何目录：
```bash
/home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug/offline_perception_debug_384_sob.sh
```

### 方法2：配置CLion运行环境变量

1. 在CLion中打开 `Run` -> `Edit Configurations...`
2. 选择你的运行配置（如 `offline_perception_debug_384_sob`）
3. 在 `Environment variables` 字段中添加：
   ```
   LD_LIBRARY_PATH=/home/youfeng/CLionProjects/05-offline_debug_fusion/deps_gcc11.3/x86/dnn_x86/lib:$LD_LIBRARY_PATH
   ```
4. 在 `Working directory` 字段中设置为项目根目录：
   ```
   /home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug
   ```
5. 点击 `Apply` 和 `OK`

### 方法3：使用原始运行脚本

在终端中运行：
```bash
cd /home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug
./run_384_sob.sh
```

## 验证

运行程序后，应该看到：
```
========== Initializing Perception Modules ==========
Project root: /home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug
[✓] Stereo matcher initialized (adaptive parameters for night mode)
[✓] Cdt-task model initialized: models/cdt_20251125_640x384.bin
[✓] DSG Night model initialized: models/dsg_multi_20260403_640x384.bin
```

如果看到模型加载错误，说明工作目录或模型路径配置不正确。

## 代码改进

程序已经添加了自动路径检测功能：
- 自动检测可执行文件位置并推导项目根目录
- 自动修正模型文件的相对路径
- 如果模型文件不存在，会给出清晰的错误提示

## 输出目录

- 图片输出：`/home/youfeng/debug/boluo/0123/20260413/dsg_multi_debug_7_205_432/`
- 点云输出：`/home/youfeng/debug/boluo/0123/20260413/dsg_pcd_debug_7_205_205_432/`

输出目录在代码中硬编码，可以在 `main()` 函数中修改 `input_dir` 变量来更改。
