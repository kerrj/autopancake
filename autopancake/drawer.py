from ur5py.ur5 import UR5Robot

class PancakeDrawer:
    def __init__(self):
        self.ur = UR5Robot(gripper=True)