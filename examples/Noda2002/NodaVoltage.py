import numpy as np
import matplotlib.pyplot as plt

# Novos pontos obtidos visualmente da nova figura
time_points_ns = np.array([0, 1.5, 3, 4.5, 6, 8, 10, 13, 16, 20, 25, 30, 35, 40])
voltage_points_V = np.array([0, 7, 20, 35, 45, 50, 53, 55, 56, 57, 58, 58.5, 59, 60])

# Converter tempo para segundos
time_points_s = time_points_ns * 1e-9

# Vetor de tempo com 416 pontos
N_points = 416
t = np.linspace(0, 40e-9, N_points)

# Interpolação linear
V = np.interp(t, time_points_s, voltage_points_V)

# Salvar o novo arquivo
voltage_path = 'noda_voltage.exc'
with open(voltage_path, 'w') as f:
    for ti, Vi in zip(t, V):
        f.write(f"{ti:.6e}\t{Vi:.6e}\n")

# Plot para conferência
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(t * 1e9, V, 'r-', linewidth=2, label='Aproximação linear atualizada')
plt.plot(time_points_ns, voltage_points_V, 'ko', label='Pontos da figura')
plt.xlabel('Tempo (ns)')
plt.ylabel('Tensão (V)')
plt.title('Forma de onda da tensão (atualizada) - Noda')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

voltage_path
