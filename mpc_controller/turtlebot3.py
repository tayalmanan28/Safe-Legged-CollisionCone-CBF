import pybullet as p
import time
import pybullet_data
p.connect(p.GUI)
offset = [0,0,0]
p.setAdditionalSearchPath(pybullet_data.getDataPath())
turtle = p.loadURDF("urdf/turtlebot.urdf",offset)
plane = p.loadURDF("plane.urdf")
p.setRealTimeSimulation(1)

for j in range (p.getNumJoints(turtle)):
	print(p.getJointInfo(turtle,j))
forward=0
turn=0
i = 0
p.setGravity(0,0,-10)
time.sleep(1./240.)
p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,force=0)
p.setJointMotorControl2(turtle,2,p.VELOCITY_CONTROL,force=0)

p.setJointMotorControl2(turtle,1,p.TORQUE_CONTROL,force=0)
p.setJointMotorControl2(turtle,2,p.TORQUE_CONTROL,force=0)
while (1):	
	#time.sleep(1./240.)
	_, _, _, T1 = p.getJointState(turtle, 1)
	_, _, _, T2 = p.getJointState(turtle, 2)
	
	print(T1, T2)
	
	p.setJointMotorControl2(turtle,1,p.TORQUE_CONTROL,force=10)
	p.setJointMotorControl2(turtle,2,p.TORQUE_CONTROL,force=10)
	
	#p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
	#p.setJointMotorControl2(turtle,2,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)
	
	#p.applyExternalTorque(turtle, 0, torqueObj = [0,0,0.01*i], flags=p.LINK_FRAME)
	#p.applyExternalTorque(turtle, 2, torqueObj = [0,100,0], flags=p.LINK_FRAME)


