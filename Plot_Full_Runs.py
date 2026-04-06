import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import savgol_filter
plot_titles = ['ER','TN&ER','Naive','TN']
x_list = []
y_list = []
std_list = []
titles = []
folder = f'Full_Run_Results/'
files = os.listdir(folder)
index = 0

while index < len(files):
    x = []
    y = []
    std = []
    filename = files[index]
    if filename.endswith('.csv'):
        with open(f'{folder}{filename}','r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                if row[1] == 'eval_timesteps':
                    titles.append(filename)
                else:
                    x.append(int(row[1]))
                    y.append(float(row[2]))
                    std.append(float(row[3]))
        x_list.append(x)
        y_list.append(y)
        std_list.append(std)
    index +=1
smoothing_window = 81
#learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing

fig, ax = plt.subplots()
ax.set_title(f'Experiment: Results for Naive, ER, TN and ER&TN implementations.')
ax1 = ax.twinx()
leg = []
for i in range(len(x_list)):
    smooth = savgol_filter(y_list[i],smoothing_window,2)
    err = std_list[i]/np.sqrt(5)

    err_smooth = savgol_filter(err,smoothing_window,2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode Returns')
    lns1 = ax.plot(x_list[i], smooth, label = f'{plot_titles[i]}')
    lns2 = ax.fill_between(x_list[i],smooth-err_smooth,smooth+err_smooth,alpha=0.2)
    leg += lns1 

labs = [l.get_label() for l in leg]    
ax.legend(leg, labs, loc=0)    
plt.savefig(f"Full_Run_Results/result_smoothing{smoothing_window}.png", dpi=300, bbox_inches='tight')
plt.show()