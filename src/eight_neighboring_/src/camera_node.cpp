#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cctype>
#include <string>
#include <thread>

class CameraNode : public rclcpp::Node {
public:
  CameraNode() : Node("camera_node") {
    device_index_ = declare_parameter<int>("device_index", 0);
    width_        = declare_parameter<int>("width", 640);
    height_       = declare_parameter<int>("height", 480);
    fps_          = declare_parameter<int>("fps", 30);
    fixed_rate_output_ = declare_parameter<bool>("fixed_rate_output", false);
    use_video_    = declare_parameter<bool>("use_video", true);
    video_path_   = declare_parameter<std::string>("video_path", "/home/mechax/lf_demo/test.mp4");
    frame_id_     = declare_parameter<std::string>("frame_id", "camera_frame");
    image_topic_  = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    pixel_format_ = declare_parameter<std::string>("pixel_format", "MJPG");

    pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic_, rclcpp::SensorDataQoS());

    if (use_video_) {
      cap_.open(video_path_);
      if (!cap_.isOpened()) {
        RCLCPP_FATAL(get_logger(), "Cannot open video file: %s", video_path_.c_str());
        throw std::runtime_error("video open failed");
      }
    } else {
      cap_.open(device_index_, cv::CAP_V4L2);
      if (!cap_.isOpened()) {
        RCLCPP_FATAL(get_logger(), "Cannot open camera index %d", device_index_);
        throw std::runtime_error("camera open failed");
      }
      const int requested_fourcc = parseFourcc(pixel_format_);
      if (requested_fourcc != 0) {
        cap_.set(cv::CAP_PROP_FOURCC, static_cast<double>(requested_fourcc));
      }
      cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
      cap_.set(cv::CAP_PROP_FPS,          fps_);
    }

    if (fixed_rate_output_) {
      const int period_ms = (fps_ > 0) ? (1000 / fps_) : 33;
      timer_ = create_wall_timer(
        std::chrono::milliseconds(period_ms),
        std::bind(&CameraNode::tick, this)
      );
    } else {
      capture_thread_ = std::thread(&CameraNode::captureLoop, this);
    }

    RCLCPP_INFO(get_logger(),
      "CameraNode started: index=%d, %dx%d, fps=%d, fixed_rate_output=%s, topic=%s",
      device_index_, width_, height_, fps_,
      fixed_rate_output_ ? "true" : "false", image_topic_.c_str());

    if (!use_video_) {
      const int actual_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
      const int actual_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
      const double actual_fps = cap_.get(cv::CAP_PROP_FPS);
      const int actual_fourcc = static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC));
      RCLCPP_INFO(
        get_logger(),
        "Camera negotiated: format=%s, %dx%d @ %.2f fps (requested format=%s)",
        fourccToString(actual_fourcc).c_str(),
        actual_w, actual_h, actual_fps, pixel_format_.c_str());
    }
  }

  ~CameraNode() override {
    running_.store(false);
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
  }

private:
  static int parseFourcc(std::string s) {
    if (s.empty()) return 0;
    for (char & c : s) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (s.size() != 4) return 0;
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
  }

  static std::string fourccToString(int fourcc) {
    std::string s(4, ' ');
    s[0] = static_cast<char>(fourcc & 0xFF);
    s[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    s[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    s[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    for (char & c : s) {
      if (!std::isprint(static_cast<unsigned char>(c))) {
        c = '?';
      }
    }
    return s;
  }

  bool readFrame(cv::Mat & frame) {
    if (!cap_.read(frame) || frame.empty()) {
      if (use_video_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(frame) || frame.empty()) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to read frame");
          return false;
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to read frame");
        return false;
      }
    }
    return true;
  }

  void publishFrame(const cv::Mat & frame) {
    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = frame_id_;

    auto msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
    pub_->publish(*msg);
  }

  void tick() {
    cv::Mat frame;
    if (!readFrame(frame)) {
      return;
    }
    publishFrame(frame);
  }

  void captureLoop() {
    const double source_fps = cap_.get(cv::CAP_PROP_FPS);
    const bool use_video_pacing = use_video_ && source_fps > 1.0;
    const auto video_period = std::chrono::duration<double>(1.0 / source_fps);

    while (rclcpp::ok() && running_.load()) {
      const auto loop_start = std::chrono::steady_clock::now();

      cv::Mat frame;
      if (!readFrame(frame)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      publishFrame(frame);

      if (use_video_pacing) {
        const auto elapsed = std::chrono::steady_clock::now() - loop_start;
        const auto remaining = video_period - elapsed;
        if (remaining > std::chrono::duration<double>::zero()) {
          std::this_thread::sleep_for(remaining);
        }
      }
    }
  }

private:
  int device_index_{0}, width_{640}, height_{480}, fps_{30};
  bool fixed_rate_output_{false};
  bool use_video_{true};
  std::string video_path_{"/home/mechax/lf_demo/test.mp4"};
  std::string frame_id_{"camera_frame"};
  std::string image_topic_{"/camera/image_raw"};
  std::string pixel_format_{"MJPG"};

  std::atomic<bool> running_{true};
  cv::VideoCapture cap_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::thread capture_thread_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraNode>());
  rclcpp::shutdown();
  return 0;
}