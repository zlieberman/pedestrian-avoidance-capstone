#include "robot/robot.h"
#include <algorithm>


TurtleBot::TurtleBot() {
  /// Publish the velocities to the robot on the navigation topic
  publishVelocity = nh.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1000);
  /// Subscribe for data from the laser sensor on the scan topic
  subscibeSensor = nh.subscribe<sensor_msgs::LaserScan>("/scan", 500, &TurtleBot::sensorCallback, this);
  obstacleDetected = false;
  ROS_INFO_STREAM("Turtlebot Initialized");
}

TurtleBot::~TurtleBot() {
  // Make sure the turtlebot stops moving
  stop();
}

void TurtleBot::start() {
  ros::Rate loop_rate(rate);

  while (ros::ok()) {
    if (obstacleDetected) {
      // Start turning the robot to avoid obstacles
      velocities.linear.x = 0.0;
      velocities.angular.z = anguarVelocity;
    } else {
      // Start moving the robot once obstacle is avoided
      velocities.angular.z = 0.0;
      velocities.linear.x = linearVelocity;
    }

    // Publish the velocities
    publishVelocity.publish(velocities);
    // Handle callback
    ros::spinOnce();
    // Make the system sleep to maintain loop rate
    loop_rate.sleep();
  }
}

void TurtleBot::sensorCallback(const sensor_msgs::LaserScan::ConstPtr& sensorData) {
  auto checkSensorData = [](float range){ return range<distanceThreshold; };
  // for now just indicate that an obstacle has been detected
  obstacleDetected = std::any_of(sensorData->ranges.begin(), sensorData->ranges.end(), checkSensorData);
}

void TurtleBot::stop() {
  // Reset linear velocity of the turtlebot
  velocities.linear.x = 0.0;
  velocities.linear.y = 0.0;
  velocities.linear.z = 0.0;
  // Reset angular velocity of the turtlebot
  velocities.angular.x = 0.0;
  velocities.angular.y = 0.0;
  velocities.angular.z = 0.0;

  publishVelocity.publish(velocities);
  ROS_INFO_STREAM("Bringing Turtlebot to a stop");
}
