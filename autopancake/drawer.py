from ur5py.ur5 import UR5Robot
from typing import List, Tuple
from autolab_core import RigidTransform
import numpy as np

class PancakeDrawer:
    
    def __init__(self, pan_center: Tuple[float,float,float]):
        self.ur = UR5Robot(gripper=True)
        self.ur.gripper.open()
        self.close_rate = 0.2 #TODO calibrate this: #mm of gripper close distance per mm of travel
        self.pan_center = pan_center
        self.draw_speed = .05#meters/s
        self.ur.gripper.set_speed(50)#TODO calibrate this based on move speed
        
    def draw_path(self, path: List[Tuple[float,float]]):
        '''
        Given a list of coords, 
        '''
        self.ur.set_tcp(RigidTransform(translation=[0,0,0.2]))
        self._goto_waypoint(path[0],asyn=False)
        for i,coord in enumerate(path[1:]):
            self._goto_waypoint(coord,asyn=True,vel = self.draw_speed)
            #calculate distance from i to i-1
            #convert to mm for the close_rate constant
            distance = 1000*np.linalg.norm(np.array(coord)-np.array(path[i-1]))
            #TODO move the gripper based on current position
            self.ur.gripper.move(distance*self.close_rate)
            #TODO wait until robot stops moving to continue
            
            
    def _goto_waypoint(self, waypoint: Tuple[float,float,float], **ur_kwargs):
        #TODO
        pass
        
    def close_on_bottle(self):
        self.ur.gripper.set_force(20)#TODO calibrate this
        self.ur.gripper.close()