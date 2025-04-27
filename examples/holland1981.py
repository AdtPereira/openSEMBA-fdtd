# conda activate semba-fdtd
# cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
# python examples/holland1981.py

import numpy as np
import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt

# Permite importar src_pyWrapper e utils mesmo rodando da raiz
sys.path.insert(0, os.getcwd())

EXCITATIONS_FOLDER = os.path.join('examplesData', 'excitations')
OUTPUTS_FOLDER = os.path.join('examplesData', 'outputs')
LOGS_FOLDER = os.path.join('examplesData', 'logs')
CASES_FOLDER = os.path.join('examplesData', 'cases')

def create_case_holland1981():
    """
    Create holland1981.fdtd.json by following exactly the structure and indentation of existing_holland1981.fdtd.json.
    Using CaseMaker where possible, and direct input when needed.
    """
    from src_pyWrapper.pyWrapper import CaseMaker  # Ajuste o import conforme seu projeto

    cm = CaseMaker()

    # --- 1. Headers ---
    cm.input['format'] = "FDTD Input file"
    cm.input['__comments'] = "1m linear antenna illuminated by a pulse : Holland, R. Finite-Difference Analysis of EMP Coupling to Thin Struts and Wires. 2000. IEEE-TEMC."

    # --- 2. General section ---
    cm.input['general'] = {
        "timeStep": 30e-12,
        "numberOfSteps": 1000
    }

    # --- 3. Boundary conditions ---
    cm.input['boundary'] = {
        "all": {
            "type": "pml",
            "layers": 6,
            "order": 2.0,
            "reflection": 0.001
        }
    }

    # --- 4. Materials ---
    cm.input['materials'] = [
        {
            "id": 1,
            "type": "wire",
            "radius": 0.02,
            "resistancePerMeter": 0.0,
            "inductancePerMeter": 0.0
        },
        {
            "id": 2,
            "type": "terminal",
            "terminations": [{"type": "open"}]
        }
    ]

    # --- 5. Mesh: Grid definition ---
    cm.input['mesh'] = {
        "grid": {
            "numberOfCells": [20, 20, 22],
            "steps": {
                "x": [0.1],
                "y": [0.1],
                "z": [0.1]
            }
        },
        "coordinates": [
            {"id": 1, "relativePosition": [11, 11, 7]},
            {"id": 2, "relativePosition": [11, 11, 12]},
            {"id": 3, "relativePosition": [11, 11, 17]},
            {"id": 4, "relativePosition": [12, 11, 17]}
        ],
        "elements": [
            {"id": 1, "type": "node", "coordinateIds": [2]},
            {"id": 2, "type": "polyline", "coordinateIds": [1, 2, 3]},
            {"id": 3, "type": "cell", "intervals": [[[1, 1, 1], [19, 19, 21]]]},
            {"id": 4, "type": "node", "coordinateIds": [4]}
        ]
    }

    # --- 6. Material Associations ---
    cm.input['materialAssociations'] = [
        {
            "name": "single_wire",
            "materialId": 1,
            "initialTerminalId": 2,
            "endTerminalId": 2,
            "elementIds": [2]
        }
    ]

    # --- 7. Sources ---
    cm.input['sources'] = [
        {
            "type": "planewave",
            "magnitudeFile": "holland.exc",
            "elementIds": [3],
            "direction": {
                "theta": 1.5708,
                "phi": 0.0
            },
            "polarization": {
                "theta": 0.0,
                "phi": 0.0
            }
        }
    ]

    # --- 8. Probes ---
    cm.input['probes'] = [
        {
            "name": "mid_point",
            "type": "wire",
            "field": "current",
            "elementIds": [1],
            "domain": {"type": "time"}
        }
    ]

    # --- 9. Export case with pretty indentation ---
    output_path = 'examplesData/cases/holland1981.fdtd.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        import json
        json.dump(cm.input, f, indent=4)  # <<< Prettify with indent=4
        f.write('\n')  # optional: ensures newline at end of file

    print(f"✅ Case 'holland1981.fdtd.json' generated successfully at {output_path}.")

def create_excitation_holland():
    """
    Create the holland.exc file, representing a double-exponential pulse.
    """

    # Parameters for the excitation
    a = 2.3e8  # decay rate 1
    b = 4.6e8  # decay rate 2
    dt = 30e-12  # time step (same as FDTD)
    n_steps = 1000  # number of steps (same as simulation steps)

    # Time vector
    time = np.linspace(0, (n_steps-1)*dt, n_steps)

    # Analytical excitation
    src = np.exp(-a * time) - np.exp(-b * time)

    # Prepare the data for saving
    data = np.column_stack((time, src))

    # Create folder if necessary
    os.makedirs(EXCITATIONS_FOLDER, exist_ok=True)

    # Save file
    output_path = os.path.join(EXCITATIONS_FOLDER, 'holland.exc')
    np.savetxt(output_path, data, fmt="%.5e", delimiter=" ")

    print(f"✅ Excitation file 'holland.exc' created successfully at {output_path}.")

def run_simulation():
    """ Run the FDTD simulation using the generated case and excitation files. """

    # --- Verifica se o executável existe ---
    executable = os.path.join("build", "bin", "semba-fdtd.exe")
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"Executable {executable} not found.")
    input_file = os.path.join(CASES_FOLDER, 'holland1981.fdtd.json')

    # --- Se o arquivo .json não existir, cria-o ---
    if not os.path.isfile(input_file):
        print(f"⚡ Input file {input_file} not found. Generating it now...")
        create_case_holland1981()

    # --- Se o arquivo holland.exc não existir, cria-o ---
    excitation_file = os.path.join(EXCITATIONS_FOLDER, 'holland.exc')
    if not os.path.isfile(excitation_file):
        print(f"⚡ Excitation file {excitation_file} not found. Generating it now...")
        create_excitation_holland()

    # --- Copiar arquivo de excitação para a raiz (fdtd/) ---
    if not os.path.isfile('holland.exc'):
        if os.path.isfile(excitation_file):
            shutil.copy2(excitation_file, '.')
        else:
            raise FileNotFoundError(f"Excitation file {excitation_file} not found.")

    # --- Executar o solver ---
    print(f"Executando: {executable} -i {input_file}")
    subprocess.run([executable, "-i", input_file], check=True)
    print("Simulação finalizada.")

    # --- Criar pastas se necessário ---
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)

    # --- Mover arquivos ---
    for file in os.listdir(CASES_FOLDER):
        if file.startswith('holland1981'):
            src = os.path.join(CASES_FOLDER, file)
            if file.endswith('.dat') or file.endswith('.h5'):
                dst = os.path.join(OUTPUTS_FOLDER, file)
                shutil.move(src, dst)
            elif file.endswith('.txt'):
                dst = os.path.join(LOGS_FOLDER, file)
                shutil.move(src, dst)

    # --- Limpar o arquivo de excitação temporário ---
    if os.path.isfile('holland.exc'):
        os.remove('holland.exc')

def check_output_files(output_folder, expected_files):
    """Confere se todos os arquivos esperados foram gerados."""
    missing_files = []
    found_files = []

    for expected_file in expected_files:
        full_path = os.path.join(output_folder, expected_file)
        if os.path.isfile(full_path):
            found_files.append(expected_file)
        else:
            missing_files.append(expected_file)

    print("\n📂 Verificação de arquivos gerados:")
    if found_files:
        print("✅ Arquivos encontrados:")
        for file in found_files:
            print(f"   - {file}")
    if missing_files:
        print("❌ Arquivos faltando:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("\n🎯 Todos os arquivos esperados foram encontrados!")

def plot_excitation(excitation_filename):
    """
    Reads and plots the excitation profile from an .exc file
    and compare with the analytical expression.
    The analytical expression is given by:
    src_ana = exp(-2.3e8 * time) - exp(-4.6e8 * time)

    Ref.: https://ieeexplore.ieee.org/document/4091427 

    Parameters
    ----------
    excitation_path : str
        Path to the excitation file (.exc).
    """
    # Build the full path automatically
    excitation_path = os.path.join(EXCITATIONS_FOLDER, excitation_filename)
    if not os.path.isfile(excitation_path):
        raise FileNotFoundError(f"Excitation file {excitation_path} not found.")

    # Read the excitation data
    data = np.loadtxt(excitation_path)
    time = data[:, 0]  # Time (seconds)
    src_num = data[:, 1]  # Excitation value

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(time * 1e9,
                src_num,
                label = r'$E(t) = e^{-2.3 \times 10^8 t} - e^{-4.6 \times 10^8 t}$',
                linestyle='-', linewidth=2)
    plt.xlabel('Time [ns]')
    plt.ylabel('Excitation amplitude')
    plt.title('Excitation Profile - holland.exc')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_output_midpoint(output_filename):
    """
    Reads and plots the FDTD output from a .dat file,
    specifically for the mid_point wire probe.

    Parameters
    ----------
    output_filename : str
        Name of the output file (e.g., 'holland1981.fdtd_mid_point_Wz_11_11_12_s2.dat').
    """
    # Build full path automatically
    output_path = os.path.join(OUTPUTS_FOLDER, output_filename)

    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Output file {output_path} not found.")

    # Read the output data
    data = np.loadtxt(output_path, skiprows=1)  # Skip header line

    time = data[:, 0]  # Time (seconds)
    current = data[:, 1]  # Current (Amperes)

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(time * 1e9, current, label='Mid-point Current', linestyle='-', linewidth=2)
    plt.xlabel('Time [ns]')
    plt.ylabel('Current [A]')
    plt.title('Mid-point Wire Current - ' + output_filename)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Executa a simulação ---
    # run_simulation()

    # --- Depois da simulação, confere os resultados ---
    expected_files = [
        'holland1981.fdtd_mid_point_Wz_11_11_12_s2.dat',
        'holland1981.fdtd_Energy.dat'
    ]
    
    # Verifica se os arquivos de saída foram gerados
    check_output_files(OUTPUTS_FOLDER, expected_files)
    
    # --- Plota os resultados ---
    print("\n📊 Analisando resultados:")
    plot_excitation('holland.exc')
    plot_output_midpoint('holland1981.fdtd_mid_point_Wz_11_11_12_s2.dat')
    print("\nAnálise de resultados concluída.")

