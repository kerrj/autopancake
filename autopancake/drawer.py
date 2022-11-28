from ur5py.ur5 import UR5Robot
from typing import List, Tuple
from autolab_core import RigidTransform
import numpy as np
import time
class PancakeDrawer:
    HOME_JOINTS = np.array([0.21088454127311707, -2.3260086218463343, -1.6002996603595179, -2.3468831221209925, 0.19129669666290283, 4.7159647941589355])
    def __init__(self, pan_center: Tuple[float,float,float]):
        self.ur = UR5Robot(gripper=True)
        self.ur.gripper_set.set_speed(0)
        self.ur.gripper_set.set_force(0)
        self.ur.gripper.move_and_wait_for_pos(0,255,255)
        self.close_rate = 0.05 #TODO calibrate this: % of gripper close distance per mm of travel
        self.pan_center = np.array(pan_center)
        self.draw_speed = .05#meters/s
        
    def draw_path(self, path: List[Tuple[float,float]]):
        '''
        Given a list of coords, 
        '''
        # self.ur.gripper_set.set_force(50)
        self.ur.gripper_set.set_force(5)
        self.ur.set_tcp(RigidTransform(translation=[0,0,0.15],rotation=RigidTransform.y_axis_rotation(-np.pi/2)))
        poses = [RigidTransform(translation=np.array(waypoint)+self.pan_center,rotation = RigidTransform.z_axis_rotation(-np.pi/2)) for waypoint in path]
        self.ur.move_pose(poses[0],vel=.3)
        self.ur.move_tcp_path(poses[1:],vels = [self.draw_speed]*len(poses[1:]),
                            accs = [1.0]*len(poses[1:]),
                            blends = [.005]*(len(poses[1:])-1)+[0],
                            asyn=True)
        last_pose = None
        cur_dist = self.ur.gripper.get_current_position()
        while not self.ur.is_stopped():
            if last_pose is None:
                last_pose = self.ur.get_pose()
                time.sleep(.1)
                continue
            cur_pose = self.ur.get_pose()
            dist_traveled = 1000*np.linalg.norm(cur_pose.translation-last_pose.translation)
            last_pose = cur_pose
            cur_dist += dist_traveled*self.close_rate*(255/500)
            print("cur_dist",cur_dist)
            self.ur.gripper.move(np.clip(cur_dist,0,255),0,20)
            time.sleep(.1)
    
    def draw_multi_paths(self, paths: List):
        start_gripper_pos = self.ur.gripper.get_current_position()
        for path in paths:
            if path.shape[-1]==2:
                path = np.concatenate([path,np.zeros((len(path),1))],axis=1)
            self.draw_path(path)
            self.ur.gripper.move(np.clip(start_gripper_pos,0,255),0,20)
        
    def home(self):
        self.ur.move_joint(self.HOME_JOINTS,vel=.2)
        
    def close_on_bottle(self):
        self.ur.gripper.move_and_wait_for_pos(255,0,0)#TODO calibrate this
