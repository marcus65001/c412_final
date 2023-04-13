#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
dt-exec roslaunch apriltag apriltag_node.launch veh:=csc22909
#dt-exec roslaunch main_control control_node.launch veh:=csc22925 stall:=4
dt-exec roslaunch parking parking_node.launch veh:=csc22909 stall:=4


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
