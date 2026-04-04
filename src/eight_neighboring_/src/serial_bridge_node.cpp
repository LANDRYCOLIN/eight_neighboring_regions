#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr uint8_t SOF = 0xA5;
constexpr uint8_t TYPE_VISION_POINT = 0x01;
constexpr uint8_t TYPE_HEARTBEAT = 0x81;
constexpr uint8_t PAYLOAD_LEN_VISION = 6;
constexpr uint8_t PAYLOAD_LEN_HEARTBEAT = 5;
constexpr uint8_t TYPE_VISION_ECHO = 0x82;
constexpr uint8_t PAYLOAD_LEN_ECHO = 6;

uint16_t crc16_ccitt(const uint8_t * data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int j = 0; j < 8; ++j) {
      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}
}

class SerialBridgeNode : public rclcpp::Node {
public:
  SerialBridgeNode() : Node("serial_bridge_node"), seq_(0) {
    serial_port_ = declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
    corner_topic_ = declare_parameter<std::string>("corner_topic", "/line/corner");
    publish_heartbeat_log_ = declare_parameter<bool>("log_heartbeat", true);
    confidence_value_ = declare_parameter<int>("confidence_value", 255);
    send_period_ms_ = declare_parameter<int>("send_period_ms", 20);

    if (!openPort()) {
      RCLCPP_FATAL(get_logger(), "Failed to open serial port %s", serial_port_.c_str());
      throw std::runtime_error("Failed to open serial port");
    }

    latest_point_ = {false, 0.0, 0.0};

    corner_sub_ = create_subscription<geometry_msgs::msg::Point>(
      corner_topic_, 10,
      std::bind(&SerialBridgeNode::onCorner, this, std::placeholders::_1));

    const auto period = std::chrono::milliseconds(std::max(5, send_period_ms_));
    timer_ = create_wall_timer(period, std::bind(&SerialBridgeNode::onTimer, this));

    running_.store(true);
    reader_thread_ = std::thread(&SerialBridgeNode::readLoop, this);

    RCLCPP_INFO(get_logger(), "Serial bridge running on %s (115200 8N1)", serial_port_.c_str());
  }

  ~SerialBridgeNode() override {
    running_.store(false);
    if (reader_thread_.joinable()) {
      reader_thread_.join();
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

private:
  struct VisionPoint {
    bool valid;
    double x;
    double y;
  };

  bool openPort() {
    fd_ = ::open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      RCLCPP_ERROR(get_logger(), "open(%s) failed: %s", serial_port_.c_str(), strerror(errno));
      return false;
    }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
      RCLCPP_ERROR(get_logger(), "tcgetattr failed: %s", strerror(errno));
      return false;
    }

    cfmakeraw(&tty);
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1; // 0.1s

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
      RCLCPP_ERROR(get_logger(), "tcsetattr failed: %s", strerror(errno));
      return false;
    }

    return true;
  }

  void onCorner(const geometry_msgs::msg::Point::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(point_mutex_);
    VisionPoint vp;
    vp.valid = std::isfinite(msg->x) && std::isfinite(msg->y);
    vp.x = msg->x;
    vp.y = msg->y;
    latest_point_ = vp;
  }

  void onTimer() {
    if (fd_ < 0) {
      return;
    }
    VisionPoint vp;
    {
      std::lock_guard<std::mutex> lock(point_mutex_);
      vp = latest_point_;
    }

    uint8_t payload[PAYLOAD_LEN_VISION] = {0};
    payload[0] = vp.valid ? 1 : 0;
    const auto clamp_norm = [](double v) {
      if (!std::isfinite(v)) return 0.0;
      return std::max(0.0, std::min(1.0, v));
    };
    uint16_t x_q = static_cast<uint16_t>(std::round(clamp_norm(vp.x) * 10000.0));
    uint16_t y_q = static_cast<uint16_t>(std::round(clamp_norm(vp.y) * 10000.0));
    payload[1] = static_cast<uint8_t>(x_q & 0xFF);
    payload[2] = static_cast<uint8_t>((x_q >> 8) & 0xFF);
    payload[3] = static_cast<uint8_t>(y_q & 0xFF);
    payload[4] = static_cast<uint8_t>((y_q >> 8) & 0xFF);
    payload[5] = static_cast<uint8_t>(std::clamp(confidence_value_, 0, 255));

    std::array<uint8_t, 1 + 1 + 1 + 1 + PAYLOAD_LEN_VISION + 2> frame{};
    frame[0] = SOF;
    frame[1] = PAYLOAD_LEN_VISION;
    frame[2] = TYPE_VISION_POINT;
    frame[3] = seq_++;
    std::memcpy(frame.data() + 4, payload, PAYLOAD_LEN_VISION);
    const uint16_t crc = crc16_ccitt(frame.data(), frame.size() - 2);
    frame[4 + PAYLOAD_LEN_VISION] = static_cast<uint8_t>(crc & 0xFF);
    frame[5 + PAYLOAD_LEN_VISION] = static_cast<uint8_t>((crc >> 8) & 0xFF);

    ssize_t written = ::write(fd_, frame.data(), frame.size());
    if (written < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Serial write failed: %s", strerror(errno));
    }
  }

  void readLoop() {
    std::vector<uint8_t> buffer;
    buffer.reserve(64);
    while (running_.load()) {
      uint8_t byte = 0;
      ssize_t n = ::read(fd_, &byte, 1);
      if (n == 1) {
        buffer.push_back(byte);
        parseBuffer(buffer);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  }

  void parseBuffer(std::vector<uint8_t> & buffer) {
    while (buffer.size() >= 4) {
      if (buffer[0] != SOF) {
        buffer.erase(buffer.begin());
        continue;
      }
      const uint8_t len = buffer[1];
      const size_t total = 1 + 1 + 1 + 1 + len + 2;
      if (buffer.size() < total) {
        return;
      }
      const uint16_t crc_calc = crc16_ccitt(buffer.data(), total - 2);
      const uint16_t crc_rx = static_cast<uint16_t>(buffer[total - 2]) |
                              (static_cast<uint16_t>(buffer[total - 1]) << 8);
      if (crc_calc != crc_rx) {
        buffer.erase(buffer.begin());
        continue;
      }
      const uint8_t type = buffer[2];
      if (type == TYPE_HEARTBEAT && len == PAYLOAD_LEN_HEARTBEAT) {
        Heartbeat hb;
        hb.mode = buffer[4];
        hb.err = buffer[5];
        hb.seq_echo = buffer[6];
        hb.counter = static_cast<uint16_t>(buffer[7]) |
                     (static_cast<uint16_t>(buffer[8]) << 8);
        if (publish_heartbeat_log_) {
          RCLCPP_INFO(get_logger(), "Heartbeat mode=%u err=%u seq=%u counter=%u",
                      hb.mode, hb.err, hb.seq_echo, hb.counter);
        }
      } else if (type == TYPE_VISION_ECHO && len == PAYLOAD_LEN_ECHO) {
        const uint8_t valid = buffer[4];
        const uint16_t x_q = static_cast<uint16_t>(buffer[5]) |
                             (static_cast<uint16_t>(buffer[6]) << 8);
        const uint16_t y_q = static_cast<uint16_t>(buffer[7]) |
                             (static_cast<uint16_t>(buffer[8]) << 8);
        const uint8_t conf = buffer[9];
        const uint8_t seq = buffer[3];

        RCLCPP_INFO(get_logger(),
          "[ECHO] seq=%u valid=%u x_q=%u y_q=%u conf=%u (x=%.4f y=%.4f)",
          seq, valid, x_q, y_q, conf, x_q / 10000.0, y_q / 10000.0);
      }
      buffer.erase(buffer.begin(), buffer.begin() + total);
    }
  }

  struct Heartbeat {
    uint8_t mode;
    uint8_t err;
    uint8_t seq_echo;
    uint16_t counter;
  };

private:
  std::string serial_port_;
  std::string corner_topic_;
  bool publish_heartbeat_log_; 
  int confidence_value_;
  int send_period_ms_;

  int fd_{-1};
  std::atomic<bool> running_{false};
  std::thread reader_thread_;

  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr corner_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex point_mutex_;
  VisionPoint latest_point_;
  uint8_t seq_;

};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SerialBridgeNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("serial_bridge_node"), "Exception: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}