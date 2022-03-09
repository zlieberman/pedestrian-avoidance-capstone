#include "robot/robot.h"

using namespace std;

int main(int argc, char* argv[]) {
  // Initialize the ROS node
  ros::init(argc, argv, "turtlebot_pedestrian_avoidance");

  TurtleBot turtleBot;
  turtleBot.start();

  return 0;
}
