import pybullet as p
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='label3.urdf', type=str)
args = parser.parse_args()

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

p.configureDebugVisualizer(1, lightPosition=(5, 0, 5), rgbBackground=(1,1,1))
obj = p.loadURDF(f"output/{args.name}", [0, 0, 0], [0,0,0,1], useFixedBase=True)
# visualization
while True:
    for jid in range(p.getNumJoints(obj)):
        ji = p.getJointInfo(obj, jid)
        joint_name = ji[1].decode('ascii')
        if "drawer" in joint_name:
            jointpos = np.random.uniform(0, 0.4)
            p.resetJointState(obj, jid, jointpos)
        elif "doorL" in joint_name:
            jointpos = np.random.uniform(-1.57, 0)
        elif "doorR" in joint_name:
            jointpos = np.random.uniform(0, 1.57)

        p.resetJointState(obj, jid, jointpos)
        time.sleep(0.2)
    # p.stepSimulation()