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
        auto_thresh_k_   = declare_parameter<double>("auto_thresh_k", 0.6); // 建议在 launch 中改为 0.85 滤除光斑
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

        // 保留旧参数防报错
        declare_parameter<int>("window_width", 160); 
        declare_parameter<int>("window_height", 40); 
        declare_parameter<int>("num_windows", 4); 
        declare_parameter<std::vector<double>>("region_weights", {0.0});
        
        show_fps_overlay_ = declare_parameter<bool>("show_fps_overlay", true);
        fps_ema_alpha_    = declare_parameter<double>("fps_ema_alpha", 0.2);
        local_debug_display_ = declare_parameter<bool>("local_debug_display", true);
        
        batch_mode_        = declare_parameter<bool>("batch_mode", false);
        input_video_path_  = declare_parameter<std::string>("input_video_path", "");
        output_video_path_ = declare_parameter<std::string>("output_video_path", "debug_batch.mp4");

        corner_pub_ = create_publisher<geometry_msgs::msg::Point>(corner_topic_, 10);
        debug_pub_  = create_publisher<sensor_msgs::msg::Image>(debug_topic_, 10);
        binary_pub_ = create_publisher<sensor_msgs::msg::Image>(binary_topic_, 10);

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_, rclcpp::SensorDataQoS(),
            std::bind(&EightNavNode::onImage, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "八邻域摩尔追踪：加入光斑垂直跨度过滤与完美 L/T 区分机制！");
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
        // 【核心：八邻域 Moore 边界追踪】
        // ==========================================
        
        int total_white_pixels = cv::countNonZero(binary);
        double global_white_ratio = (double)total_white_pixels / (roi_rect.area());
        bool is_dead_end = (global_white_ratio < 0.015); 

        cv::Point center_seed(-1, -1);
        if (!is_dead_end) {
            int mid_x = w / 2;
            for (int y = roi_h - 5; y >= 0; y--) { 
                for (int x_offset = 0; x_offset < w / 2; x_offset += 2) {
                    if (binary.at<uchar>(y, mid_x + x_offset) > 0) { center_seed = cv::Point(mid_x + x_offset, y); break; }
                    if (binary.at<uchar>(y, mid_x - x_offset) > 0) { center_seed = cv::Point(mid_x - x_offset, y); break; }
                }
                if (center_seed.x != -1) break;
            }
        }
        if (center_seed.x == -1) is_dead_end = true;

        int current_x = w / 2;
        int current_y = roi_h / 2; 

        if (!is_dead_end) {
            cv::Point left_seed = center_seed;
            while(left_seed.x > 0 && binary.at<uchar>(left_seed.y, left_seed.x - 1) > 0) left_seed.x--;
            
            cv::Point right_seed = center_seed;
            while(right_seed.x < w - 1 && binary.at<uchar>(right_seed.y, right_seed.x + 1) > 0) right_seed.x++;
            
            int base_width = std::max(20, right_seed.x - left_seed.x);

            std::vector<cv::Point> left_edge = traceBoundary(binary, left_seed, 4, true);
            std::vector<cv::Point> right_edge = traceBoundary(binary, right_seed, 0, false);

            // 【抗光斑强化】：计算轮廓的垂直高度跨度 (Vertical Span)
            int min_y = roi_h;
            for(const auto& p : left_edge) min_y = std::min(min_y, p.y);
            for(const auto& p : right_edge) min_y = std::min(min_y, p.y);
            int vertical_span = center_seed.y - min_y;

            // 真正的赛道一定会向上延伸很长。如果垂直跨度小于 ROI 高度的 30%（比如是个圆圈光斑），直接干掉！
            if (left_edge.size() + right_edge.size() < 80 || vertical_span < roi_h * 0.30) {
                is_dead_end = true;
            } else {
                int min_left_x = w;
                int max_right_x = 0;
                int max_track_width = 0;
                
                std::vector<int> left_b(roi_h, w - 1); 
                std::vector<int> right_b(roi_h, 0);    
                
                for(const auto& p : left_edge) {
                    left_b[p.y] = std::min(left_b[p.y], p.x); 
                    min_left_x = std::min(min_left_x, p.x); 
                    cv::circle(debug_vis, cv::Point(p.x, p.y + roi_y), 2, cv::Scalar(0, 0, 255), -1); 
                }
                for(const auto& p : right_edge) {
                    right_b[p.y] = std::max(right_b[p.y], p.x); 
                    max_right_x = std::max(max_right_x, p.x); 
                    cv::circle(debug_vis, cv::Point(p.x, p.y + roi_y), 2, cv::Scalar(255, 0, 0), -1); 
                }

                for(int y = 0; y < roi_h; y++) {
                    if (left_b[y] < w - 1 && right_b[y] > 0) {
                        max_track_width = std::max(max_track_width, right_b[y] - left_b[y]);
                    }
                }

                // 【降维打击分类器：通过膨胀和双向展宽区分 L / T / 岔路】
                bool is_branched = (max_track_width > w * 0.35); // 赛道物理宽度是否发生剧烈膨胀（排除了普通 L 弯）
                
                bool is_wide_left = ((center_seed.x - min_left_x) > w * 0.25);
                bool is_wide_right = ((max_right_x - center_seed.x) > w * 0.25);

                int target_y = roi_h / 2; 
                while(target_y < roi_h - 1 && (left_b[target_y] == w - 1 || right_b[target_y] == 0)) {
                    target_y++; 
                }
                
                if (target_y >= roi_h - 1) {
                    current_x = center_seed.x;
                    current_y = center_seed.y;
                } else {
                    current_y = target_y;
                    
                    if (is_branched) {
                        if (is_wide_left && is_wide_right) {
                            // T-Junction (双向起飞)
                            current_x = right_b[target_y] - base_width / 2; 
                            cv::putText(debug_vis, "T-JUNCTION -> TURN RIGHT", cv::Point(w / 2 - 200, 100), 
                                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
                        } else if (is_wide_left && !is_wide_right) {
                            // 仅左岔路 -> 忽略左侧，死咬右边缘直行
                            current_x = right_b[target_y] - base_width / 2; 
                            cv::putText(debug_vis, "IGNORE LEFT -> GO STRAIGHT", cv::Point(w / 2 - 250, 100), 
                                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 3);
                        } else if (!is_wide_left && is_wide_right) {
                            // 仅右岔路 (右 T 型) -> 进入右转逻辑
                            current_x = right_b[target_y] - base_width / 2; 
                            cv::putText(debug_vis, "RIGHT BRANCH -> TURN RIGHT", cv::Point(w / 2 - 250, 100), 
                                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
                        } else {
                            current_x = (left_b[target_y] + right_b[target_y]) / 2;
                        }
                    } else {
                        // 正常的直线或 L 型弯：取赛道中点，平滑巡线
                        current_x = (left_b[target_y] + right_b[target_y]) / 2;
                    }
                }

                geometry_msgs::msg::Point p;
                p.x = static_cast<double>(current_x) / w;
                p.y = static_cast<double>(current_y + roi_y) / h;
                corner_pub_->publish(p);

                cv::drawMarker(debug_vis, cv::Point(current_x, current_y + roi_y), 
                               cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 30, 3);
            }
        } 
        
        if (is_dead_end) {
            cv::putText(debug_vis, "DEAD END", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
        }

        if (show_fps_overlay_ && fps_value_ > 0.0) {
            std::ostringstream fps_ss;
            fps_ss << std::fixed << std::setprecision(1) << "FPS: " << fps_value_;
            cv::putText(debug_vis, fps_ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat small_debug;
        cv::resize(debug_vis, small_debug, cv::Size(), 0.5, 0.5);
        auto debug_msg = cv_bridge::CvImage(header, "bgr8", small_debug).toImageMsg();
        debug_pub_->publish(*debug_msg);

        if (local_debug_display_) {
            cv::imshow("TRUE 8-NEIGHBORHOOD TRACING", debug_vis);
            cv::waitKey(1);
        }
        
        return debug_vis;
    }

    std::vector<cv::Point> traceBoundary(const cv::Mat& bin, cv::Point start, int start_bg_dir, bool clockwise) const {
        std::vector<cv::Point> edge;
        cv::Point curr = start;
        int current_bg_dir = start_bg_dir;
        
        const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        for(int i = 0; i < 2000; i++) { 
            edge.push_back(curr);
            
            if (curr.y <= 0 || curr.x <= 0 || curr.x >= bin.cols - 1) break;
            if (i > 50 && curr.y >= bin.rows - 1) break;
            
            bool found = false;
            for(int j = 0; j < 8; j++) {
                int check_dir = clockwise ? (current_bg_dir + j) % 8 : (current_bg_dir - j + 8) % 8;
                cv::Point p = curr + cv::Point(dx[check_dir], dy[check_dir]);
                
                if (p.x < 0 || p.x >= bin.cols || p.y < 0 || p.y >= bin.rows) continue;
                
                if (bin.at<uchar>(p) > 0) { 
                    int bg_idx_before_found = clockwise ? (check_dir - 1 + 8) % 8 : (check_dir + 1) % 8;
                    cv::Point bg_pixel = curr + cv::Point(dx[bg_idx_before_found], dy[bg_idx_before_found]);
                    
                    curr = p; 
                    cv::Point rel_bg = bg_pixel - curr;
                    
                    for(int k = 0; k < 8; k++) {
                        if(dx[k] == rel_bg.x && dy[k] == rel_bg.y) {
                            current_bg_dir = k; 
                            break;
                        }
                    }
                    found = true;
                    break;
                }
            }
            if (!found) break; 
        }
        return edge;
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

public:
    void runBatchProcessing() {
        if (input_video_path_.empty()) return;
        cv::VideoCapture cap(input_video_path_);
        if (!cap.isOpened()) return;

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        cv::VideoWriter writer(output_video_path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        RCLCPP_INFO(get_logger(), "开始批处理: %dx%d @ %.2f FPS", width, height, fps);

        cv::Mat frame;
        while (rclcpp::ok() && cap.read(frame)) {
            std_msgs::msg::Header header; header.stamp = this->now();
            cv::Mat processed = processFrame(frame, header);
            writer.write(processed);
        }
        RCLCPP_INFO(get_logger(), "批处理完成！");
    }

private:
    void applyBorderMask(cv::Mat &mask) const {
        if (mask.empty()) return;
        const int margin = std::max(0, border_margin_px_);
        if (margin == 0) return;
        const int margin_x = std::min(margin, mask.cols / 2 - 1);
        const int margin_y = std::min(margin, mask.rows / 2 - 1);
        if (margin_y > 0) { mask.rowRange(0, margin_y).setTo(0); mask.rowRange(mask.rows - margin_y, mask.rows).setTo(0); }
        if (margin_x > 0) { mask.colRange(0, margin_x).setTo(0); mask.colRange(mask.cols - margin_x, mask.cols).setTo(0); }
    }

    void updateRealtimeFps() {
        auto now = std::chrono::steady_clock::now();
        if (!fps_initialized_) { last_frame_tp_ = now; fps_initialized_ = true; return; }
        std::chrono::duration<double> dt = now - last_frame_tp_;
        last_frame_tp_ = now;
        if (dt.count() > 1e-6) fps_value_ = fps_ema_alpha_ * (1.0 / dt.count()) + (1.0 - fps_ema_alpha_) * fps_value_; 
    }

    std::string image_topic_, corner_topic_, debug_topic_, binary_topic_;
    double roi_ratio_, auto_thresh_k_;
    bool auto_threshold_;
    int threshold_, auto_thresh_min_, auto_thresh_max_;
    int blur_ksize_, bilateral_d_;
    double bilateral_sigma_color_, bilateral_sigma_space_;
    int morph_ksize_, close_ksize_, open_ksize_, border_margin_px_;
    
    double fps_ema_alpha_, fps_value_{0.0};
    bool show_fps_overlay_, fps_initialized_{false}, local_debug_display_;
    std::chrono::steady_clock::time_point last_frame_tp_;

    bool batch_mode_;              
    std::string input_video_path_; 
    std::string output_video_path_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr corner_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr binary_pub_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EightNavNode>();
    bool is_batch = false; 
    node->get_parameter("batch_mode", is_batch);
    if (is_batch) node->runBatchProcessing();
    else rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}