#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/point.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

class EightNavNode : public rclcpp::Node {
public:
    EightNavNode() : Node("eight_nav_node") {
        image_topic_  = declare_parameter<std::string>("image_topic", "/camera/image_raw");
        corner_topic_ = declare_parameter<std::string>("corner_topic", "/line/corner");
        debug_topic_  = declare_parameter<std::string>("debug_topic", "/line/debug_image");
        binary_topic_ = declare_parameter<std::string>("binary_topic", "/line/binary_image");
        
        roi_ratio_       = declare_parameter<double>("roi_ratio", 0.6); 
        auto_threshold_  = declare_parameter<bool>("auto_threshold", true);
        threshold_       = declare_parameter<int>("threshold", 210);
        auto_thresh_k_   = declare_parameter<double>("auto_thresh_k", 0.6);
        auto_thresh_min_ = declare_parameter<int>("auto_thresh_min", 160);
        auto_thresh_max_ = declare_parameter<int>("auto_thresh_max", 235);
        
        blur_ksize_            = declare_parameter<int>("blur_ksize", 5);
        bilateral_d_           = declare_parameter<int>("bilateral_d", 5);
        bilateral_sigma_color_ = declare_parameter<double>("bilateral_sigma_color", 25.0);
        bilateral_sigma_space_ = declare_parameter<double>("bilateral_sigma_space", 25.0);
        
        morph_ksize_ = declare_parameter<int>("morph_ksize", 5);
        close_ksize_ = declare_parameter<int>("close_ksize", 9);
        open_ksize_  = declare_parameter<int>("open_ksize", 5);
        border_margin_px_ = declare_parameter<int>("border_margin_px", 0);

        // ==========================================
        // 【核心修改 1：将正方形窗口拆分为宽扁矩形】
        // ==========================================
        window_width_  = declare_parameter<int>("window_width", 160); 
        window_height_ = declare_parameter<int>("window_height", 40); 
        num_windows_   = declare_parameter<int>("num_windows", 4); 
        region_weights_ = declare_parameter<std::vector<double>>("region_weights", {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0}); // 默认左四为0，右四为1 
        
        show_fps_overlay_ = declare_parameter<bool>("show_fps_overlay", true);
        fps_ema_alpha_    = declare_parameter<double>("fps_ema_alpha", 0.2);
        
        // 【核心修改 2：开启原生 OpenCV 显示后门】
        local_debug_display_ = declare_parameter<bool>("local_debug_display", true);
        
        // ======= 新增：从 Launch 获取批处理配置 =======
        batch_mode_        = declare_parameter<bool>("batch_mode", false);
        input_video_path_  = declare_parameter<std::string>("input_video_path", "");
        output_video_path_ = declare_parameter<std::string>("output_video_path", "debug_batch.mp4");
        // ===========================================

        corner_pub_ = create_publisher<geometry_msgs::msg::Point>(corner_topic_, 10);
        debug_pub_  = create_publisher<sensor_msgs::msg::Image>(debug_topic_, 10);
        binary_pub_ = create_publisher<sensor_msgs::msg::Image>(binary_topic_, 10);

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_, rclcpp::SensorDataQoS(),
            std::bind(&EightNavNode::onImage, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "八邻域巡线启动. 模式: 宽扁窗口 + 原生视窗输出.");
    }

private:
    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        updateRealtimeFps(); 
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            const cv::Mat& frame = cv_ptr->image;
            if (frame.empty()) return;
            processFrame(frame, msg->header);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "异常: %s", e.what());
        }
    }

    cv::Mat processFrame(const cv::Mat& frame, const std_msgs::msg::Header& header) {
        int w = frame.cols;
        int h = frame.rows;

        int roi_y = static_cast<int>(h * (1.0 - roi_ratio_));
        int roi_h = h - roi_y;
        if (roi_y < 0 || roi_h <= 0) return frame;
        cv::Rect roi_rect(0, roi_y, w, roi_h);
        cv::Mat roi = frame(roi_rect);

        cv::Mat gray, binary;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        applyPreFilters(gray);

        double threshold_value = static_cast<double>(threshold_);
        if (auto_threshold_) {
            cv::Scalar mean, stddev;
            cv::meanStdDev(gray, mean, stddev);
            threshold_value = mean[0] + auto_thresh_k_ * stddev[0];
            threshold_value = std::clamp(threshold_value, static_cast<double>(auto_thresh_min_), static_cast<double>(auto_thresh_max_));
        }
        cv::threshold(gray, binary, threshold_value, 255, cv::THRESH_BINARY);

        int k = std::max(1, morph_ksize_);
        if (k % 2 == 0) k += 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {k, k});
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

        if (close_ksize_ > 1) {
            int ck = std::max(3, close_ksize_ | 1);
            cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, {ck, ck});
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, close_kernel);
        }
        
        if (open_ksize_ > 1) {
            int ok = std::max(3, open_ksize_ | 1);
            cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, {ok, ok});
            cv::morphologyEx(binary, binary, cv::MORPH_OPEN, open_kernel);
        }

        applyBorderMask(binary);

        cv::Mat small_binary;
        cv::resize(binary, small_binary, cv::Size(), 0.5, 0.5);
        auto bin_msg = cv_bridge::CvImage(header, "mono8", small_binary).toImageMsg();
        binary_pub_->publish(*bin_msg);

        cv::Mat debug_vis = frame.clone();
        cv::rectangle(debug_vis, roi_rect, cv::Scalar(255, 255, 0), 2);

        // ==========================================
        // 【核心逻辑重构：动态找点 + 宽扁窗口防吞噬】
        // ==========================================
        int current_y = roi_h - window_height_ / 2;
        int current_x = w / 2;

        // 1. 在底部扫一条横线，找寻真实的白线起点，防止锚定在画面中央
        int search_y_start = std::max(0, roi_h - window_height_);
        cv::Mat bottom_strip = binary.rowRange(search_y_start, roi_h);
        cv::Moments M_bottom = cv::moments(bottom_strip, true);
        if (M_bottom.m00 > 0) {
            current_x = static_cast<int>(M_bottom.m10 / M_bottom.m00);
        }

        bool is_dead_end = false;
        bool is_junction = false; // 【新增】用来记录当前帧是否发现了路口

        for (int i = 0; i < num_windows_; ++i) {
            int x_min = std::max(0, current_x - window_width_ / 2);
            int x_max = std::min(w - 1, current_x + window_width_ / 2);
            int y_min = std::max(0, current_y - window_height_ / 2);
            int y_max = std::min(roi_h - 1, current_y + window_height_ / 2);

            cv::Rect window_rect(x_min, y_min, x_max - x_min, y_max - y_min);
            if (window_rect.area() <= 0) break;

            cv::Mat window_roi = binary(window_rect);
            bool hits_side = false;
            // 传入 15 像素作为防抖阈值，小于 15 像素的毛刺一律不当分支
            int branches = analyzeTopology(window_roi, hits_side, 15);

            cv::Moments M = cv::moments(window_roi, true);
            double white_ratio = M.m00 / window_rect.area(); 
            
            if (branches <= 1 && i == 0) {
                if (white_ratio < 0.1) {
                    is_dead_end = true; 
                    break;
                }
            }

            // 【新增】默认框是绿色 (B, G, R)
            cv::Scalar box_color(0, 255, 0); 

           // ======= 修改判决条件 =======
            // 现在要判定为路口，必须满足 4 把锁：
            // 1. 分支 >= 3 (已经防抖)
            // 2. 不是最底下的头两层锚点 (i > 1)
            // 3. 白色面积足够大 (white_ratio > 0.15)
            // 4. 白线必须怼到了窗口的左边或右边 (hits_side)
            if (branches >= 3 && i > 1 && white_ratio > 0.15 && hits_side) {
                
                // --- 权重控制转向逻辑保持不变 ---
                cv::Mat weighted_roi;
                window_roi.convertTo(weighted_roi, CV_32F);

                int strip_w = window_roi.cols / 8;
                for (int j = 0; j < 8; ++j) {
                    int x_start = j * strip_w;
                    int x_end = (j == 7) ? window_roi.cols : (j + 1) * strip_w;
                    
                    // 应用权重：将该区域像素值乘以对应的权重系数
                    cv::Mat strip = weighted_roi.colRange(x_start, x_end);
                    strip *= region_weights_[j];
                }

                // 使用加权后的图像重新计算质心
                M = cv::moments(weighted_roi, false); 
                
                // 标记为路口并变红
                is_junction = true;                
                box_color = cv::Scalar(0, 0, 255); 
            }

            if (M.m00 > 0) {
                current_x = x_min + static_cast<int>(M.m10 / M.m00);
                current_y = y_min + static_cast<int>(M.m01 / M.m00);
            } else {
                current_y -= window_height_;
            }

            // 画框时使用动态颜色
            cv::rectangle(debug_vis, cv::Rect(x_min, y_min + roi_y, x_max - x_min, y_max - y_min), box_color, 2);
            current_y -= window_height_;
        }

        if (!is_dead_end) {
            geometry_msgs::msg::Point p;
            p.x = static_cast<double>(current_x) / w;
            p.y = static_cast<double>(current_y + roi_y) / h;
            corner_pub_->publish(p);

            // ==========================================
            // 【核心显性绘制：输出直观结果】
            // ==========================================
            
            // 1. 画一个紫色的十字准星，代表最终发给底盘的引导点
            cv::drawMarker(debug_vis, cv::Point(current_x, current_y + roi_y + window_height_), 
                           cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 30, 3);

            // 2. 如果标记了路口，在屏幕上打大字提示
            if (is_junction) {
                cv::putText(debug_vis, "JUNCTION -> TURN RIGHT", cv::Point(w / 2 - 200, 100), 
                            cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            }
        } else {
            cv::putText(debug_vis, "DEAD END", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
        }

        if (show_fps_overlay_ && fps_value_ > 0.0) {
            std::ostringstream fps_ss;
            fps_ss << std::fixed << std::setprecision(1) << "FPS: " << fps_value_;
            cv::putText(debug_vis, fps_ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 4);
            cv::putText(debug_vis, fps_ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat small_debug;
        cv::resize(debug_vis, small_debug, cv::Size(), 0.5, 0.5);
        auto debug_msg = cv_bridge::CvImage(header, "bgr8", small_debug).toImageMsg();
        debug_pub_->publish(*debug_msg);

        // ==========================================
        // 【原生 OpenCV 窗口】
        // ==========================================
        if (local_debug_display_) {
            cv::imshow("NATIVE DEBUG VIEW (NO RQT LAG)", debug_vis);
            cv::waitKey(1);
        }
        
        return debug_vis; // 新增：返回渲染后的图像
    }

    void applyPreFilters(cv::Mat &gray) const {
        if (gray.empty()) return;
        if (blur_ksize_ > 1) {
            int ksize = std::max(3, blur_ksize_ | 1);
            cv::GaussianBlur(gray, gray, cv::Size(ksize, ksize), 0);
        }
        if (bilateral_d_ > 0) {
            cv::Mat temp;
            cv::bilateralFilter(gray, temp, bilateral_d_,
                                std::max(1.0, bilateral_sigma_color_),
                                std::max(1.0, bilateral_sigma_space_));
            gray = temp;
        }
    }

    // ======= 新增：脱机逐帧处理逻辑 =======
public:
    void runBatchProcessing() {
        if (input_video_path_.empty()) {
            RCLCPP_ERROR(get_logger(), "未指定输入视频路径！");
            return;
        }

        cv::VideoCapture cap(input_video_path_);
        if (!cap.isOpened()) {
            RCLCPP_ERROR(get_logger(), "无法打开视频文件: %s", input_video_path_.c_str());
            return;
        }

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        cv::VideoWriter writer(output_video_path_, 
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                               fps, cv::Size(width, height));

        RCLCPP_INFO(get_logger(), "开始批处理: %dx%d @ %.2f FPS", width, height, fps);

        cv::Mat frame;
        while (rclcpp::ok() && cap.read(frame)) {
            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "batch_frame";

            // 逐帧调用算法，不经过 ROS 话题，绝不丢帧
            cv::Mat processed = processFrame(frame, header);
            writer.write(processed);
        }
        RCLCPP_INFO(get_logger(), "批处理完成！文件保存至: %s", output_video_path_.c_str());
    }

    void applyBorderMask(cv::Mat &mask) const {
        if (mask.empty()) return;
        const int margin = std::max(0, border_margin_px_);
        if (margin == 0) return;
        
        const int max_margin_x = std::max(0, mask.cols / 2 - 1);
        const int max_margin_y = std::max(0, mask.rows / 2 - 1);
        const int margin_x = std::min(margin, max_margin_x);
        const int margin_y = std::min(margin, max_margin_y);

        if (margin_y > 0) {
            mask.rowRange(0, margin_y).setTo(0);
            mask.rowRange(mask.rows - margin_y, mask.rows).setTo(0);
        }
        if (margin_x > 0) {
            mask.colRange(0, margin_x).setTo(0);
            mask.colRange(mask.cols - margin_x, mask.cols).setTo(0);
        }
    }

    // 【核心重构】：具备防抖与侧边检测的拓扑分析器
    int analyzeTopology(const cv::Mat& roi, bool& hits_side, int min_branch_width = 15) const {
        int w = roi.cols, h = roi.rows;
        if (w < 2 || h < 2) {
            hits_side = false;
            return 0;
        }

        // 1. 提取周长一维数组
        std::vector<uchar> perimeter;
        perimeter.reserve(2 * w + 2 * h);
        for (int x = 0; x < w; ++x) perimeter.push_back(roi.at<uchar>(0, x));
        for (int y = 1; y < h; ++y) perimeter.push_back(roi.at<uchar>(y, w - 1));
        for (int x = w - 2; x >= 0; --x) perimeter.push_back(roi.at<uchar>(h - 1, x));
        for (int y = h - 2; y > 0; --y) perimeter.push_back(roi.at<uchar>(y, 0));

        // 2. 状态机防抖：滤除小于 min_branch_width 的细碎锯齿噪点
        std::vector<uchar> compressed;
        int current_color = perimeter[0];
        int run_length = 0;

        for (uchar p : perimeter) {
            if (p == current_color) {
                run_length++;
            } else {
                if (run_length >= min_branch_width) {
                    compressed.push_back(current_color);
                }
                current_color = p;
                run_length = 1;
            }
        }
        if (run_length >= min_branch_width) {
            compressed.push_back(current_color);
        }

        // 首尾闭环处理
        if (compressed.size() > 1 && compressed.front() == compressed.back()) {
            compressed.pop_back();
        }

        // 3. 侧边溢出校验：检查窗口极左和极右是否被粗壮白线穿透
        int left_white = cv::countNonZero(roi.col(0));
        int right_white = cv::countNonZero(roi.col(w - 1));
        hits_side = (left_white >= min_branch_width || right_white >= min_branch_width);

        return compressed.size() / 2; // 返回真实的粗壮分支数
    }

    void updateRealtimeFps() {
        auto now = std::chrono::steady_clock::now();
        if (!fps_initialized_) { last_frame_tp_ = now; fps_initialized_ = true; return; }
        std::chrono::duration<double> dt = now - last_frame_tp_;
        last_frame_tp_ = now;
        if (dt.count() > 1e-6) {
            fps_value_ = fps_ema_alpha_ * (1.0 / dt.count()) + (1.0 - fps_ema_alpha_) * fps_value_;
        }
    }

    std::string image_topic_, corner_topic_, debug_topic_, binary_topic_;
    std::vector<double> region_weights_;
    double roi_ratio_, auto_thresh_k_;
    bool auto_threshold_;
    int threshold_, auto_thresh_min_, auto_thresh_max_;
    int blur_ksize_, bilateral_d_;
    double bilateral_sigma_color_, bilateral_sigma_space_;
    int morph_ksize_, close_ksize_, open_ksize_, border_margin_px_;
    
    // 改成了独立的宽和高
    int window_width_, window_height_, num_windows_;
    
    double fps_ema_alpha_, fps_value_{0.0};
    bool show_fps_overlay_, fps_initialized_{false}, local_debug_display_;
    std::chrono::steady_clock::time_point last_frame_tp_;

    // ======= 新增：批处理模式相关变量 =======
    bool batch_mode_;              // 是否开启批处理模式
    std::string input_video_path_; // 输入原视频路径
    std::string output_video_path_;// 输出渲染视频路径
    // =====================================

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr corner_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr binary_pub_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EightNavNode>();

    // 获取是否开启批处理的参数
    bool is_batch;
    node->get_parameter("batch_mode", is_batch);

    if (is_batch) {
        // 模式 A：离线批处理，处理完即退出
        node->runBatchProcessing();
    } else {
        // 模式 B：在线实时巡线
        rclcpp::spin(node);
    }

    rclcpp::shutdown();
    return 0;
}