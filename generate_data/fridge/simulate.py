from pyBulletSimRecorder import PyBulletRecorder
import time
import pybullet as p
import numpy as np
import os

def simulate(dir, joint, joint_limit, rot_max, rot_min, pt):
    # Can alternatively pass in p.DIRECT 
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -10, physicsClientId=client) 

    urdf_file = os.path.join(dir, "mobility.urdf")

    object = p.loadURDF(urdf_file, [0,0,0.1], (0, 0, 0.5, 0.5), useFixedBase=1)
    box = p.loadURDF("/home/evelyn/Desktop/partnet-reasoning/shapes/small_sphere4.urdf", pt, (0, 0, 0.5, 0.5), useFixedBase=1)

    for i in range (4):
        p.changeDynamics(object, i+1, lateralFriction=0.001, spinningFriction=0.005)

    numJoints = p.getNumJoints(object)
    for j in range (numJoints):
        p.setJointMotorControl2(object,j, p.VELOCITY_CONTROL,force=1)
        p.resetJointState(object, j, 0.0)

    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-60, cameraPitch=-15, cameraTargetPosition=[0,0,0])
    recorder = PyBulletRecorder()

    recorder.register_object(object, urdf_file)
    recorder.register_object(box, "/home/evelyn/Desktop/partnet-reasoning/shapes/small_sphere4.urdf")

    joint_range = rot_max - joint_limit

    for t in range (500):
        if not(- 0.00628 * t <= -rot_max + joint_limit): 
            p.resetJointState(object, joint, -0.00628 * t)
        # p.resetJointState(object, joint, rot_max - joint_range * t / 500)

        if (t+1) % 50 == 0:
            recorder.add_keyframe()

    # recorder.add_keyframe()
    
    recorder.save(os.path.join(dir, "demo.pkl"))

    p.disconnect()