import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 摄像头节点
    camera_node = Node(
        package='eight_neighboring_regions',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'device_index': 0,
            'width': 640,
            'height': 480,
            'fps': 30,
            'fixed_rate_output': False,
            'use_video': True, # 测试视频时改为 True
            'video_path': '/home/causin/eight_neighboring_regions/test.mp4',
            'frame_id': 'camera_frame',
            'image_topic': '/camera/image_raw',
            'pixel_format': 'MJPG'
        }]
    )

    # 2. 核心八邻域巡线节点
    eight_nav_node = Node(
        package='eight_neighboring_regions',
        executable='eight_nav_node',
        name='eight_nav_node',
        output='screen',
        parameters=[{
            'roi_ratio': 0.5,
            
            # --- 阈值与上下限截断 ---
            'auto_threshold': True,
            'auto_thresh_k': 0.85,
            'auto_thresh_min': 165, # 在暗光死角的绝对兜底线
            'auto_thresh_max': 225, # 在强光直射下的最高阈值
            
            # --- 降噪与保边 ---
            'blur_ksize': 7,
            'bilateral_d': 0,       # 设为大于0开启双边滤波，专治地板反光纹理
            
            # --- 形态学流水线 ---
            'morph_ksize': 7,       # 第一刀：抹掉微小星点
            'close_ksize': 13,       # 第二刀：补齐光斑导致的线段断裂
            'open_ksize': 7,        # 第三刀：刮平多余毛刺
            
            # --- 边缘物理遮挡 ---
            'border_margin_px': 25,  # 设为 20 即可把赛道边缘杂物物理抹黑
            
            # --- 其他 ---
            'window_width': 360,    # 足够宽，包住白线
            'window_height': 35,    # 足够薄，应对弯道
            'num_windows': 6,
            'local_debug_display': True,
            'region_weights': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.2, 1.5]
        }]
    )

    # 3. 串口通信节点
    serial_bridge_node = Node(
        package='eight_neighboring_regions',
        executable='serial_bridge_node',
        name='serial_bridge_node',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'corner_topic': '/line/corner',
            'log_heartbeat': True,
            'confidence_value': 255,
            'send_period_ms': 20
        }]
    )

    # ==========================
    # GUI 可视化弹窗模块 
    # ==========================

    # 4. 仅启动一个 rqt_image_view 窗口
    # 默认显示效果图，可通过界面左上角下拉菜单切换 /camera/image_raw 和 /line/binary_image
    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view' 
    )

    return LaunchDescription([
        camera_node,
        eight_nav_node,
        serial_bridge_node,
        rqt_image_view_node  # 部署到无显示器的实车时，将此行注释掉即可
    ])