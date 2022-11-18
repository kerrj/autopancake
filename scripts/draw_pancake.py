from autopancake.drawer import PancakeDrawer
import numpy as np
import matplotlib.pyplot as plt
from autopancake.planner import generate_circle_traj

PAN_CENTER = (.65,0,.0)

if __name__ == '__main__':
    drawer = PancakeDrawer(PAN_CENTER)

    waypoints = generate_circle_traj(.1,0.001,0.1)

    # plt.plot(waypoints[:,0],waypoints[:,1])
    # plt.show()

    #home the arm
    drawer.home()
    
    # #accept the bottle
    input("Press enter to close around the bottle")
    drawer.close_on_bottle()
    print("finished closing")
    
    #execute the drawing
    drawer.draw_path(waypoints)