# sphere_rcs.py
# Local: C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd\examples\sphere_rcs\
# Executar a partir da raiz do projeto: fdtd/
# Exemplo de execução: python examples/sphere_rcs.py

import numpy as np
import subprocess
import os
import sys
import shutil
import matplotlib.pyplot as plt

# Permite importar src_pyWrapper e utils mesmo rodando da raiz
sys.path.insert(0, os.getcwd())

from src_pyWrapper.pyWrapper import CaseMaker, Probe

EXCITATIONS_FOLDER = os.path.join('examplesData', 'excitations')
OUTPUTS_FOLDER = os.path.join('examplesData', 'outputs')
CASES_FOLDER = os.path.join('examplesData', 'cases')

# --- 1) Gera o caso sphere_rcs ---
def generate_sphere_rcs_case():
    cm = CaseMaker()

    cm.setNumberOfTimeSteps(100)  # Definido 100 passos para evitar erro
    cm.setAllBoundaries("mur")
    cm.setGridFromVTK("testData/geometries/sphere.grid.vtp")

    sphereId = cm.addCellElementsFromVTK("testData/geometries/buggy_sphere.str.vtp")
    pecId = cm.addPECMaterial()
    cm.addMaterialAssociation(pecId, [sphereId])

    planewaveBoxId = cm.addCellElementBox([[-75.0, -75.0, -75.0], [75.0, 75.0, 75.0]])
    direction = {"theta": np.pi/2, "phi": 0.0}
    polarization = {"theta": np.pi/2, "phi": np.pi/2}

    dt = 1e-12
    w0 = 0.1e-9
    t0 = 10 * w0
    t = np.arange(0, t0 + 20*w0, dt)
    data = np.empty((len(t), 2))
    data[:, 0] = t
    data[:, 1] = np.exp(-np.power(t - t0, 2) / w0**2)

    # --- Salva gauss.exc ---
    os.makedirs(EXCITATIONS_FOLDER, exist_ok=True)
    gauss_exc_path = os.path.join(EXCITATIONS_FOLDER, 'gauss.exc')
    np.savetxt(gauss_exc_path, data)

    cm.addPlanewaveSource(planewaveBoxId, gauss_exc_path, direction, polarization)

    pointProbeNodeId = cm.addNodeElement([-65.0, 0.0, 0.0])
    cm.addPointProbe(pointProbeNodeId, name="front")

    n2ffBoxId = cm.addCellElementBox([[-85.0, -85.0, -85.0], [85.0, 85.0, 85.0]])
    theta = {"initial": np.pi/2, "final": np.pi/2, "step": 0.0}
    phi = {"initial": np.pi, "final": np.pi, "step": 0.0}
    domain = {
        "type": "frequency",
        "initialFrequency": 10e6,
        "finalFrequency": 1e9,
        "numberOfFrequencies": 10,
        "frequencySpacing": "logarithmic"
    }
    cm.addFarFieldProbe(n2ffBoxId, "n2ff", theta, phi, domain)

    os.makedirs(CASES_FOLDER, exist_ok=True)
    json_path = os.path.join(CASES_FOLDER, 'sphere_rcs')
    cm.exportCase(json_path)

    print("Caso sphere_rcs gerado com sucesso.")

# --- 2) Roda a simulação usando semba-fdtd ---
def run_simulation():
    executable = os.path.join("build", "bin", "semba-fdtd.exe")
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"Executable {executable} not found.")

    input_file = os.path.join(CASES_FOLDER, 'sphere_rcs.fdtd.json')

    print(f"Executando: {executable} -i {input_file}")
    subprocess.run([executable, "-i", input_file], check=True)
    print("Simulação finalizada.")

    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    for ext in (".txt", ".dat", ".h5"):
        for file in os.listdir():
            if file.startswith('sphere_rcs') and file.endswith(ext):
                shutil.move(file, os.path.join(OUTPUTS_FOLDER, file))

# --- 3) Carrega o resultado e plota o RCS ---
def analyze_results():
    far_field_filename = os.path.join(OUTPUTS_FOLDER, 'sphere_rcs.fdtd_n2ff_Wtheta_phi.dat')

    if not os.path.isfile(far_field_filename):
        raise FileNotFoundError(f"Arquivo de resultado {far_field_filename} não encontrado.")

    far_field_probe = Probe(far_field_filename)

    theta = far_field_probe.data['theta']
    Es = far_field_probe.data['Wtheta']
    Ei = 1.0

    rcs = 4 * np.pi * np.abs(Es / Ei)**2
    rcs_db = 10 * np.log10(rcs)

    plt.figure(figsize=(8,5))
    plt.plot(np.degrees(theta), rcs_db, 'o-')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title('Radar Cross Section (RCS) vs Theta')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUTS_FOLDER, 'sphere_rcs_RCS_plot.png'))
    plt.show()

if __name__ == "__main__":
    generate_sphere_rcs_case()
    run_simulation()
    analyze_results()
    print("Análise de resultados concluída.")
