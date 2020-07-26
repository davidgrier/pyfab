import matplotlib.pyplot as plt
import json 
import numpy as np
from matplotlib import cm

with open('trajectories.json', 'r') as f:
    data = json.load(f)
trajs = [np.asarray(traj) for traj in list(data.values()) ]

with open('paths.json', 'r') as f:
    data = json.load(f)
paths = [np.asarray(path) for path in list(data.values()) ]

# for traj in trajs:
#     print(np.shape(traj))
# #     print(traj[0])
# #     print(traj[:][0])
    
#     print(np.shape(traj[:, 0]))
    
# for path in paths:
#     print(np.shape(path))
#     print(path[:, 0])
labels = ['x', 'y', 'z']
for coord in range(3):
    plt.subplot(2, 2, coord+1)
#     map(lambda traj: plt.plot(traj[:, coord], color='blue'), trajs)
#     map(lambda path: plt.plot(path[:, coord], color='red'), paths)

    for traj in trajs: plt.plot(traj[:, coord], color='blue')
    for path in paths: plt.plot(path[:, coord], color='red')

    label = labels.pop(0)
    plt.xlabel('Time (frames)')
    plt.title(label+' (pixels)')


plt.subplot(2, 2, 4)
n = max(len(traj) for traj in trajs)
cmap = cm.get_cmap('Blues')
for traj in trajs: plt.scatter(traj[:, 0], traj[:, 1], marker='s', c=cmap( np.arange(len(traj))/float(n) ))

n = max(len(path) for path in paths)    
cmap = cm.get_cmap('Reds')
for path in paths: plt.scatter(path[:, 0], path[:, 1], c=cmap( np.arange(len(path))/float(n) ))    

plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')

plt.suptitle('Saved Trajectories (blue) and Real-Time Paths (red)')
plt.show()

