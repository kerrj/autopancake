from autopancake.drawer import PancakeDrawer
PAN_CENTER = (0,-.4,-.15)

s = .05
TRAJ = [(s,s,0),(s,-s,0),(-s,-s,0),(-s,s,0),(s,s,0)]

if __name__ == '__main__':
    drawer = PancakeDrawer(PAN_CENTER)
    
    #home the arm
    drawer.home()
    
    #accept the bottle
    input("Press enter to close around the bottle")
    drawer.close_on_bottle()
    
    #execute the drawing
    drawer.draw_path(TRAJ)