import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("Tous les imports fonctionnent !")
plt.plot([1,2,3], [4,5,6])
plt.savefig('test.png')
print("Graphique généré avec succès")