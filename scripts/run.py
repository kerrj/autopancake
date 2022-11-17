import numpy as np
import matplotlib.pyplot as plt

def pol2cart(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return [x, y]

def generate_circle_traj(center,radius,dr,dtheta):
    wp_num = int(radius/dr)
    r_cor = np.linspace(0,radius,wp_num)

    theta_cor = np.arange(0,dtheta*wp_num,dtheta)
    theta_cor = theta_cor%(2*np.pi)

    #draw a big circle at the end with constant radius
    last_wp_theta = np.arange(0,2*np.pi,dtheta) + theta_cor[-1]
    last_wp_r = np.ones_like(last_wp_theta)*radius

    r_cor = np.concatenate([r_cor,last_wp_r],0)
    theta_cor = np.concatenate([theta_cor,last_wp_theta],0)

    waypoints = []
    for r,theta in zip(r_cor,theta_cor):
        waypoints.append(pol2cart(r,theta))

    waypoints = np.array(waypoints) + center
    return waypoints


waypoints = generate_circle_traj([1,1],2,0.001,0.01)

plt.plot(waypoints[:,0],waypoints[:,1])
plt.show()

