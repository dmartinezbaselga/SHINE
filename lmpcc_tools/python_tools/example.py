from ros_visuals import *
import rospy

rospy.init_node("abc")
marker_publisher = ROSMarkerPublisher("/baseline/visuals", 10)
rate = rospy.Rate(5)

while True:

    cylinder = marker_publisher.get_arrow()
    cylinder.set_color(0)
    cylinder.set_scale(0.5, 0.15, 0.15)

    pose = Pose()

    cylinder.add_marker(pose, add_orientation=True)

    marker_publisher.publish()
    rate.sleep()
