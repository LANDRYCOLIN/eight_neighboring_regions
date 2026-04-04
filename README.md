# Eight Neighboring Regions (八邻域视觉巡线) 🚗👁️

基于 ROS 2 和 OpenCV 开发的工业级视觉巡线节点。采用**自适应阈值 + 宽扁滑动窗口拓扑分析 + 形态学门控**技术，完美解决虚拟机环境下的图像撕裂、强弱光照突变、赛道噪点干扰以及粗线吞噬等工程痛点。

## ✨ 核心特性 (Key Features)

- **⚡ 虚拟机零延迟视窗 (VM-Optimized Native UI)**: 彻底抛弃 RQT 压缩图传，底层内置 `cv::imshow` 渲染通道，拒绝 UDP 丢包撕裂，在虚拟机下稳定输出丝滑 30 FPS。
- **🛡️ 极限抗扰滤波 (ASF Pipeline)**: 采用“开运算-闭运算-开运算”交替序贯滤波，配合上下限截断的自适应阈值，无惧赛道高光反光与阴影死角。
- **📏 宽扁滑动窗口 (Wide-Flat Sliding Windows)**: 打破传统正方形窗口限制，采用宽扁矩形切片，完美防止近端“粗线吞噬”，保持远端极高弯道敏锐度。
- **🧠 形态学防噪门控 (Morphological Mass Gating)**: 融合周长跳变拓扑与像素面积双重检验。无视地板缝隙与孤立噪点，精准识别 T型/十字 岔路口，并内置智能右倾引导。
- **🛑 死胡同防御 (Dead-End Detection)**: 底部 ROI 丢失或白色占比低于 10% 时，即刻触发 `DEAD END` 报警，防止小车盲目冲撞。

## 🗂️ 节点接口 (Node Interface)

### 📥 Subscriptions (订阅)
- `/camera/image_raw` (`sensor_msgs/msg/Image`) : 原始 RGB 图像输入。

### 📤 Publishers (发布)
- `/line/corner` (`geometry_msgs/msg/Point`) : 巡线目标点归一化坐标 $(x, y \in [0, 1])$。
- `/line/debug_image` (`sensor_msgs/msg/Image`) : 带有绿框、准星和文字的调试图像。
- `/line/binary_image` (`sensor_msgs/msg/Image`) : 预处理后的二值化掩码图。

## ⚙️ 核心参数 (Parameters)

| 参数名 | 类型 | 默认值 | 物理含义 / 调参指南 |
| :--- | :---: | :---: | :--- |
| `roi_ratio` | double | 0.6 | 画面截取比例 (0.6表示只看画面底部 60% 的区域) |
| `auto_threshold` | bool | true | 是否开启自适应二值化 |
| `auto_thresh_k` | double | 0.6 | 选拔严格度。越大要求白线越亮，越小容错率越高 |
| `auto_thresh_min` | int | 160 | 全黑死角下的绝对阈值兜底，防全屏误检 |
| `morph_ksize` | int | 5 | 第一道开运算核：刮除孤立的小星点噪点 |
| `close_ksize` | int | 9 | 闭运算核：焊接白线内部因反光或阴影造成的断裂 |
| `window_width` | int | 300 | 滑动窗口宽度：必须足够宽，保证能看到白线两侧的黑色地板 |
| `window_height` | int | 40 | 滑动窗口高度：薄切片，对弯道和路口的反应更敏锐 |
| `num_windows` | int | 7 | 滑动窗口数量：决定了视野能看向多远的赛道前方 |
| `local_debug_display`| bool | true | **黑科技**：强制开启 OpenCV 原生窗口，解决虚拟机 RQT 卡顿 |

## 🚀 快速启动 (Quick Start)

1. **编译节点**
```bash
colcon build --packages-select eight_neighboring_regions --symlink-install
source install/setup.bash
```
2. **启动 Launch 文件**
```bash
ros2 launch eight_neighboring_regions run_vision_tracking.launch.py
```
3. **效果调优**
启动后，会弹出名为 NATIVE DEBUG VIEW (NO RQT LAG) 的丝滑调试窗口。
- 绿色框表示：正常直线/弯道追踪。
- 红色框及 JUNCTION 提示表示：触发路口识别并已执行右转偏航引导。
- 紫色准星：当前发送给底层 PID 控制器的绝对目标锚点。
