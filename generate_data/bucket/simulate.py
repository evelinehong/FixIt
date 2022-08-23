from pyBulletSimRecorder import PyBulletRecorder
import time
import pybullet as p
import numpy as np
import os

def simulate(dir, hand_pos, ground, xmax, xmin, ymax, ymin, zmax, func, gui=False):
    # Can alternatively pass in p.DIRECT 
    if gui:
        client = p.connect(p.GUI)
    else:
        client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -10, physicsClientId=client) 

    urdf_file = os.path.join(dir, "mobility_vhacd.urdf")

    object = p.loadURDF(urdf_file, [0,0,0], (0, 0, 0.5, 0.5), useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION)
    p.changeVisualShape(object, 0, rgbaColor=[1, 0.49411764705882355, 0, 0.5 ])
    p.changeDynamics(object, 0, mass=1)

    sphere_urdf = "./assets/small_sphere.urdf"

    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf", [0,0,ground])
    p.changeDynamics(planeId, 0, lateralFriction=0.00000000000001, spinningFriction=0.000000000001)

    for i in range (4):
        p.changeDynamics(object, i+1, lateralFriction=0.001, spinningFriction=0.005)

    numJoints = p.getNumJoints(object)
    for j in range (numJoints):
        p.setJointMotorControl2(object,j, p.VELOCITY_CONTROL,force=1)

    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-60, cameraPitch=-15, cameraTargetPosition=[0,0,0])
    recorder = PyBulletRecorder()

    recorder.register_object(object, urdf_file)
    

    spheres = []
    for t in range(1000):
        if t % 150 == 0:
            position, orientation = p.getBasePositionAndOrientation(object)
            p.resetBasePositionAndOrientation(object,  position, orientation)

            sphere = p.loadURDF(sphere_urdf, [0,0,0.5], (0,0,0,0.5))
            p.changeDynamics(sphere, 0, mass=0.1)
            spheres.append(sphere)

            position, orientation = p.getBasePositionAndOrientation(object)
            p.resetBasePositionAndOrientation(object,  position, orientation)

            sphere = p.loadURDF(sphere_urdf, [0,0,1.0], (0,0,0,0.5))
            p.changeDynamics(sphere, 0, mass=0.1)
            spheres.append(sphere)

            position, orientation = p.getBasePositionAndOrientation(object)
            p.resetBasePositionAndOrientation(object,  position, orientation)

            sphere = p.loadURDF(sphere_urdf, [0,0,1.5], (0,0,0,0.5))
            p.changeDynamics(sphere, 0, mass=0.1)
            spheres.append(sphere)

            position, orientation = p.getBasePositionAndOrientation(object)
            p.resetBasePositionAndOrientation(object,  position, orientation)

        p.stepSimulation()

    for t in range(100):
        p.stepSimulation()

    original_positions = []
    spheres2 = []
    for sphere in spheres:
        position, _ = p.getBasePositionAndOrientation(sphere)

        

        if position[2] > zmax + 0.05 or position[0] > xmax or position[0] < xmin or position[1] > ymax or position[1] < ymin or (position[0] ** 2 + position[1] ** 2) > xmax**2:
            p.removeBody(sphere)
            continue

        original_positions.append(position)
        spheres2.append(sphere)
        recorder.register_object(sphere, sphere_urdf)

    keep = True
    hand_urdf = "./assets/large_sphere2.urdf"
    hand = p.loadURDF(hand_urdf, hand_pos, (0,0,0,0.5), useFixedBase=1)
    p.changeDynamics(hand, 0, mass=10000)
    recorder.register_object(hand, hand_urdf)

    for t in range (5000):
        position, orientation = p.getBasePositionAndOrientation(hand)
        position = [position[0], position[1], position[2] + 0.0005]
        p.resetBasePositionAndOrientation(hand,  position, orientation)

        if t % 200 == 0:
            recorder.add_keyframe()
        p.stepSimulation()

    mal = 0
    for (i,sphere) in enumerate(spheres2):
        position, _ = p.getBasePositionAndOrientation(sphere)
        if position[2] <= original_positions[i][2]:
            mal += 1

    position, _ = p.getBasePositionAndOrientation(object)

    if (position[0] < -2 or position[0] > 2 or position[1] < -2 or position[1] > 2):
        keep = False

    if mal > 0 and func == 'functional': keep = False
    if mal < 2 and func == 'malfunctional': keep = False
        
    recorder.add_keyframe()
    
    recorder.save(os.path.join(dir, "demo.pkl"))

    p.disconnect()
    return keep