import numpy as np
import matplotlib.pyplot as plt

# ================================
# Leitura do arquivo
# ================================

file_path = "Noda2002a.fdtd_Wire probe_Wz_12_12_2_s1.dat"

# Carregar os dados ignorando a primeira linha (cabeçalho)
data = np.loadtxt(file_path, skiprows=1)

# Separar as colunas
t = data[:, 0]    # tempo em segundos
Wz = data[:, 1]   # componente Wz em A/m

# ================================
# Plotar Wz vs Tempo
# ================================

plt.figure(figsize=(8, 4))
plt.plot(t * 1e9, Wz, 'r-', linewidth=2)
plt.xlabel('Tempo (ns)')
plt.ylabel('Wz (A/m)')
plt.title('Componente Wz vs Tempo')
plt.xlim(-10, 40)
plt.ylim(-0.1, 0.3)
plt.grid(False)
plt.tight_layout()
plt.show()
