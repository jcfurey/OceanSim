"""Example ROS2 bringup for an OceanSim vehicle.

OceanSim's headless runner (standalone/oceansim_ros2.py) publishes the platform
robot description on /robot_description (latched) and the articulation's joints
on /joint_states. This launch wires up the standard consumers:

  * robot_state_publisher -- given the SAME URDF, it listens to /joint_states and
    broadcasts the moving TF tree.
  * rviz2 -- loads the model from the latched /robot_description topic and shows
    the robot + sensor data.

Run OceanSim first, e.g.:

    ./scripts/run_oceansim_ros2.sh --platform deeptrekker_revolution \
        --urdf /assets/DeepTrekker/revolution.urdf --publish-static-tf

then:

    ros2 launch scripts/oceansim_bringup.launch.py urdf:=/assets/DeepTrekker/revolution.urdf

In RViz add a RobotModel display and set its "Description Topic" to
/robot_description, and a TF display. use_sim_time is on so stamps line up with
OceanSim's /clock.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    # Read the URDF in Python rather than ParameterValue(Command(['cat ', urdf])).
    # Command resolves to a single string that launch shlex-splits, so a path
    # with a space would tokenize wrong and leave robot_description empty.
    urdf_path = LaunchConfiguration("urdf").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    with open(urdf_path) as f:
        robot_description = f.read()
    return [
        # robot_state_publisher: URDF in as a parameter, /joint_states in (from
        # OceanSim), TF out. Remap ITS /robot_description output away so OceanSim's
        # latched TRANSIENT_LOCAL /robot_description is the single owner (rsp's TF
        # is driven by the parameter, not the topic, so this is non-breaking).
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
            remappings=[("robot_description", "robot_state_publisher/robot_description")],
        ),
        # RViz loads the model from OceanSim's latched /robot_description topic.
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "urdf",
            description="Path to the robot URDF -- the SAME file OceanSim imports "
                        "/ publishes, so robot_state_publisher's TF matches."),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        OpaqueFunction(function=_setup),
    ])
