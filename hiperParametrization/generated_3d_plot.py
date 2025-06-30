import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# 1. Ler dados
df = pd.read_csv("resultados_pareto_3objetivos.csv")

# 2. Objetivos
cdor = df['obj1_cdor'].values  # MAXIMIZAR
caer = df['obj3_caer'].values  # MINIMIZAR
time = df['obj5_time'].values  # MINIMIZAR

# 3. Inverter CDOR para cálculo da frente de Pareto
cdor_inv = -cdor
objetivos = np.vstack((cdor_inv, caer, time)).T

# 4. Encontrar frente de Pareto
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

pareto_mask = is_pareto_efficient(objetivos)

# 5. Filtrar pontos
x = cdor[pareto_mask]  # eixo X
y = caer[pareto_mask]  # eixo Y
z = time[pareto_mask]  # eixo Z

# 6. Criar grade para interpolação
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 7. Interpolar Z sobre a grade
zi = griddata((x, y), z, (xi, yi), method='cubic')

# 8. Plotar superfície
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='Spectral', edgecolor='none', alpha=0.9)

ax.set_xlabel('obj1_cdor (max)')
ax.set_ylabel('obj3_caer (min)')
ax.set_zlabel('obj5_time (min)')
ax.set_title('Superfície Interpolada da Frente de Pareto')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
