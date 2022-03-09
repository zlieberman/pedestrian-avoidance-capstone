#ifndef ROBOT_H_
#define ROBOT_H_

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Twist.h"


// Robot base class
class Robot {
  private:
    /// Define the main access point to communications with the ROS system
    ros::NodeHandle nh;
    /// Define a publisher object with topic name and buffer size of messages
    /// Make sure the listener is subscribed to the same topic name
    ros::Publisher publishVelocity;
    /// Define a subscriber object with topic name and buffer size of messages
    /// Make sure you have subscribed to the correct topic
    ros::Subscriber subscibeSensor;
    /// Initialize publishing rate
    const int rate = 2;
    /// Define twist object to publish velocities
    geometry_msgs::Twist velocities;

  public:

    /**
    * @brief Starts running the bot
    * @return void
    */
    void start();

    /**
    * @brief Callback function for subscriber
    * @param msg data from LaserScan node
    * @return void
    */
    void sensorCallback(const sensor_msgs::LaserScan::ConstPtr& msg);

    /**
    * @brief Resets velocities of the bot
    * @return void
    */
    void stop();
    
};


// Derived class of Robot representing TurtleBot
class TurtleBot: public Robot {
  private:
    /// Initialize linear velocity with 0.2m/s
    const float linearVelocity = 0.2;
    /// Initialize angular velocity with 30degrees/s
    const float anguarVelocity = 0.52;
    /// Initalize safe distance as 1.2m
    const float distanceThreshold = 1.2;
    bool obstacleDetected;

  public:
    /**
    * @brief Constructor for the class
    */
    TurtleBot();

    /**
    * @brief Destructor for the class
    */
    ~TurtleBot();

};

#endif // INCLUDE_OBSTACLE_AVOIDANCE_H_
