from autopancake.drawer import PancakeDrawer
import numpy as np
import matplotlib.pyplot as plt
from autopancake.planner import generate_circle_traj
import argparse
import pickle as pkl
PAN_CENTER = (-.13,-.64, .02)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input waypoints file")
    parser.add_argument("-s", "--scale", type=float, default=0.08,help="scale")
    args = parser.parse_args()

    drawer = PancakeDrawer(PAN_CENTER)
    if args.input is None:
        waypoints = generate_circle_traj(.1,0.001,0.1)
        all_waypoints = [waypoints for i in range(2)]
    else:
        with open(args.input,'rb') as f:
            all_waypoints = pkl.load(f)
        #normalize the traj
        all_waypoints = [
            np.concatenate([args.scale * wp, np.zeros((len(wp), 1))], axis=1)
            for wp in all_waypoints]
        
    #home the arm
    drawer.home()
    
    # #accept the bottle
    input("Press enter to close around the bottle")
    drawer.close_on_bottle(65)
    print("finished closing")
    
    #execute the drawing
    # for wp in all_waypoints:
    #     input("Press enter to do next")
    #     drawer.draw_path(wp)

    # drawer.draw_multi_paths([all_waypoints[0]] + [all_waypoints[-1]])
    drawer.draw_multi_paths(all_waypoints)

