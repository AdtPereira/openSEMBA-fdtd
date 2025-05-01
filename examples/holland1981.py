# conda activate semba-fdtd
# cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
# build\bin\semba-fdtd -i testData\input_examples\holland1981.fdtd.json
# python examples/holland1981.py

import os
import sys
import json
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
# Permite que o Python encontre módulos locais que estão na raiz do projeto openSEMBA/fdtd.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import CaseMaker

EXCITATIONS_FOLDER = os.path.join('examplesData', 'excitations')
OUTPUTS_FOLDER = os.path.join('examplesData', 'outputs')
LOGS_FOLDER = os.path.join('examplesData', 'logs')
CASES_FOLDER = os.path.join('examplesData', 'cases')

def create_case_holland1981():
    """
    Create holland1981.fdtd.json by following exactly the structure and indentation of existing_holland1981.fdtd.json.
    Using CaseMaker where possible, and direct input when needed.
    """
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
            "radius": 0.04,
            "resistancePerMeter": 0.0,
            "inductancePerMeter": 0.0
        },
        {
            "id": 2,
            "type": "terminal",
            "terminations": [{"type": "open"}]
        },
        {
            "id": 3,
            "type": "free_space",
            "relativePermittivity": 1.0,
            "relativeMagneticPermeability": 1.0,
            "conductivity": 0.0
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
        },
        {
            "name": "air_domain",
            "materialId": 3,
            "elementIds": [3]
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
                "theta": 3.1416,
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
    """ Run the FDTD simulation using the generated case and excitation files,
        gerando também o VTK da geometria diretamente em examplesData/cases.
    """
    import os
    import shutil
    import subprocess

    # --- 1) Definições iniciais ---
    cwd_root    = os.getcwd()
    rel_exe     = os.path.join("build", "bin", "semba-fdtd.exe")
    executable  = os.path.abspath(rel_exe)
    cases_dir   = os.path.join('examplesData', 'cases')
    logs_dir    = os.path.join('examplesData', 'logs')
    outputs_dir = os.path.join('examplesData', 'outputs')

    # --- 2) Validações ---
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"Executable {executable} not found.")

    json_name = 'holland1981.fdtd.json'
    json_src  = os.path.join(cases_dir, json_name)
    exc_src   = os.path.join('examplesData', 'excitations', 'holland.exc')

    if not os.path.isfile(json_src):
        print(f"⚡ Input JSON {json_src} não encontrado. Gerando…")
        create_case_holland1981()

    if not os.path.isfile(exc_src):
        print(f"⚡ Excitation {exc_src} não encontrado. Gerando…")
        create_excitation_holland()

    # --- 3) Prep de pastas e arquivos ---
    os.makedirs(cases_dir,    exist_ok=True)
    os.makedirs(outputs_dir,   exist_ok=True)
    os.makedirs(logs_dir,      exist_ok=True)

    # copia só a excitação, o JSON já está lá
    shutil.copy2(exc_src, cases_dir)

    # --- 4) Roda o solver dentro de cases_dir ---
    os.chdir(cases_dir)
    try:
        print(f"Executando: {executable} -mapvtk -i {json_name}")
        subprocess.run([executable, "-mapvtk", "-i", json_name], check=True)
    finally:
        os.chdir(cwd_root)

    # --- 5) Move resultados para outputs/logs ---
    for fname in os.listdir(cases_dir):
        if fname.startswith('holland1981') and fname.endswith(('.dat', '.h5', '.vtk')):
            shutil.move(os.path.join(cases_dir, fname),
                        os.path.join(outputs_dir, fname))
        elif fname.startswith('holland1981') and fname.endswith('.txt'):
            shutil.move(os.path.join(cases_dir, fname),
                        os.path.join(logs_dir, fname))

    print("Simulação finalizada e arquivos movidos para outputs e logs.")

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
    plt.figure(figsize=(6,5))
    plt.plot(time * 1e9, current, label='Mid-point Current', linestyle='-', linewidth=2)
    plt.xlabel('Time [ns]')
    plt.ylabel('Current [A]')
    plt.title('Mid-point Wire Current - ' + output_filename)
    plt.grid(True)
    plt.xlim(0, 32)
    plt.ylim(-8e-4, 8e-4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def export_mesh(case_json_path:  str = None,
                                  output_vtm_path: str = None,
                                  origin: tuple = (0.0, 0.0, 0.0)):
    """
    Gera um único arquivo .vtm (MultiBlock) contendo:
      - A grade (UniformGrid/ImageData)
      - O fio fino (PolyData)

    Parâmetros
    ----------
    case_json_path : str, optional
        Caminho para o JSON do CaseMaker. Se None, usa:
        CASES_FOLDER/holland1981.fdtd.json
    output_vtm_path : str, optional
        Caminho completo de saída para o arquivo .vtm combinado. Se None, usa:
        OUTPUTS_FOLDER/holland1981_combined.vtm
    origin : tuple, optional
        Origem (x, y, z) da malha, padrão (0.0,0.0,0.0).
    """
    
    # Define paths padrão se não fornecidos
    if case_json_path is None:
        case_json_path = os.path.join(CASES_FOLDER, 'holland1981.fdtd.json')
    if output_vtm_path is None:
        output_vtm_path = os.path.join(OUTPUTS_FOLDER, 'holland1981.vtm')

    # 1) Carrega JSON do caso
    with open(case_json_path, 'r') as f:
        case = json.load(f)
    mesh = case.get('mesh') or {}

    # 2) Extrai número de células e espaçamentos
    ncx, ncy, ncz = mesh['grid']['numberOfCells']
    dx = mesh['grid']['steps']['x'][0]
    dy = mesh['grid']['steps']['y'][0]
    dz = mesh['grid']['steps']['z'][0]
    dims = (ncx + 1, ncy + 1, ncz + 1)

    # 3) Cria a grade (UniformGrid ou ImageData)
    if hasattr(pv, 'UniformGrid'):
        grid = pv.UniformGrid(dims=dims, spacing=(dx, dy, dz), origin=origin)
    else:
        grid = pv.ImageData(dimensions=dims, spacing=(dx, dy, dz), origin=origin)

    # 4) Reconstrói o fio fino (polyline)
    coords = mesh.get('coordinates', [])
    coord_map = {c['id']: np.array(c['relativePosition']) * np.array([dx, dy, dz])
                 for c in coords}
    wire = None
    for elem in mesh.get('elements', []):
        if elem.get('type') == 'polyline':
            ids = elem['coordinateIds']
            pts = np.vstack([coord_map[i] for i in ids if i in coord_map])
            if pts.shape[0] >= 2:
                n = pts.shape[0]
                # conectividade VTK: [n, 0,1,2,...]
                lines = np.hstack(([n], np.arange(n)))
                wire = pv.PolyData(pts, lines)
            break

    # 5) Monta MultiBlock
    blocks = {'mesh': grid}
    if wire is not None:
        blocks['wire'] = wire
    mb = pv.MultiBlock(blocks)

    # 6) Salva em .vtm (extensão obrigatória)
    os.makedirs(os.path.dirname(output_vtm_path) or '.', exist_ok=True)
    if not output_vtm_path.lower().endswith('.vtm'):
        raise ValueError("O output_vtm_path deve terminar com '.vtm'.")
    mb.save(output_vtm_path)
    print(f"✅ Arquivo combinado salvo em: {output_vtm_path}")

    # 7) Limpeza
    del mb, grid, wire

if __name__ == "__main__":
    # --- Executa a simulação ---
    run_simulation()

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

    export_mesh()