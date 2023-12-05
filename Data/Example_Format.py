import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = np.random.normal(0,5,100)
y = np.random.normal(0,1,100)

plt.style.use("seaborn-talk")
plt.figure(figsize=(12,9))
plt.title("Example Format", size=22)
plt.ylabel("Percent Return",size=16)
plt.xlabel("Random Variable",size=16)
plt.scatter(x,y,color="darkgreen",edgecolor='b')
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.show()
