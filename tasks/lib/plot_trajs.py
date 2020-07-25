import matplotlib.pyplot as plt
import json 
import numpy as np
from matplotlib import cm


with open('trajectories.json', 'r') as f:
    data = json.load(f)
trajs = data.values()

with open('paths.json', 'r') as f:
    data = json.load(f)
paths = data.values()
   
n = [len(traj) for traj in trajs]
print(n)
nframes = max(n)
print(nframes)
cmap = cm.get_cmap('Blues')
for traj in trajs:
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    plt.plot(x, y, color='Blue')
    plt.scatter(x, y, c=cmap(np.arange(len(x))/float(nframes)))

n = [len(path) for path in paths]
print(n)
nframes = max(n)
print(nframes)
cmap = cm.get_cmap('Reds')
for path in paths:
    x = [point[0] for point in path]
    y = [point[1] for point in path]
    plt.plot(x, y, color='Red')
    plt.scatter(x, y, c=cmap(np.arange(len(x))/float(nframes)))
    
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Encircle (stepSize=0.5 -> nframes=296)')
plt.show()

plt.subplot(2, 1, 1)
plt.title('x')
for path in paths:
	plt.plot([point[0] for point in path], color='red')
for traj in trajs:
	plt.plot([point[0] for point in traj], color='blue')



plt.subplot(2, 1, 2)
plt.title('x')
for path in paths:
	plt.plot([point[1] for point in path], color='red')
for traj in trajs:
	plt.plot([point[1] for point in traj], color='blue')

plt.suptitle('Given Trajectories (blue) vs Actual Trajectories (red)')
plt.show()
