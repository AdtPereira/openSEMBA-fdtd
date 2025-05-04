'''
    Plane wave in a box example.
    This example shows how to run a simulation of a plane wave in a
    box using the openSEMBA FDTD solver.

    cmd commands:
    conda activate semba-fdtd
    cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
    python examples/plane_wave.py

    flowchart TD
    A[Início de run_simulation()] --> B[Verifica se SEMBA_EXE existe]
    B -->|OK| C[Verifica se JSON_FILE existe]
    C -->|OK| D[Salva cwd_root = os.getcwd()]
    D --> E[Muda diretório para CASES_FOLDER]
    E --> F{Try bloco}
    F --> G[Cria objeto solver]
    G --> H[solver.cleanUp()]
    H --> I[solver.run()]
    I --> J{Finally bloco}
    J --> K[Volta para cwd_root (diretório original)]
    K --> L[Copia arquivos de probe para OUTPUTS_DIR]
    L --> M[Retorna o dicionário probes]

'''
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import * # pylint: disable=unused-import,wrong-import-position

# Define o diretório atual (cwd) como o diretório raiz do projeto
# Isso é útil para garantir que os caminhos relativos funcionem corretamente.
# CWD_ROOT = C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
CWD_ROOT = os.getcwd()

# Define o caminho absoluto para o executável semba-fdtd.exe
# Diretório onde estão JSON e .exc
SEMBA_EXE           = os.path.abspath(os.path.join('build','bin','semba-fdtd.exe'))
CASES_FOLDER        = os.path.join('examplesData','cases')
EXCITATIONS_FOLDER  = os.path.join('examplesData','excitations')
OUTPUTS_FOLDER      = os.path.join('examplesData','outputs', 'plane_wave')
JSON_FILE           = os.path.join(CASES_FOLDER,'pw-in-box.fdtd.json')
ABS_OUTPUTS_FOLDER  = os.path.abspath(OUTPUTS_FOLDER)
LOGS_FOLDER         = os.path.join(CWD_ROOT,'examplesData','logs')

# Cria os diretórios de saída e logs, se não existirem
os.makedirs(ABS_OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(EXCITATIONS_FOLDER, exist_ok=True)

def make_case_pw():
    """
    Create 'pw-in-box.fdtd.json' using the CaseMaker class, replicating the structure
    of the original JSON reference file.
    """

    cm = CaseMaker()

    # 1. Header
    cm.input["format"] = "FDTD Input file"
    cm.input["__comments"] = "Plane wave passing through an empty box."

    # 2. General section
    cm.input["general"] = {
        "timeStep": 0.05e-9,
        "numberOfSteps": 400
    }

    # 3. Boundary conditions
    # cm.input["boundary"] = {
    #     "all": {"type": "mur"}
    # }
    cm.input['boundary'] = {
        "all": {
            "type": "pml",
            "layers": 6,
            "order": 2.0,
            "reflection": 0.001
        }
    }

    # 4. Mesh: grid + coordinates + elements
    cm.input["mesh"] = {
        "grid": {
            "numberOfCells": [6, 6, 6],
            "steps": {
                "x": [0.01],
                "y": [0.01],
                "z": [0.01]
            }
        },
        "coordinates": [
            {"id": 1, "relativePosition": [3, 3, 1]},
            {"id": 2, "relativePosition": [3, 3, 3]},
            {"id": 3, "relativePosition": [3, 3, 5]}
        ],
        "elements": [
            {"id": 1, "type": "node", "coordinateIds": [1]},
            {"id": 2, "type": "node", "coordinateIds": [2]},
            {"id": 3, "type": "node", "coordinateIds": [3]},
            {"id": 4, "type": "cell", "name": "pw-box", "intervals": [[[2, 2, 2], [5, 5, 5]]]}
        ]
    }

    # 5. Source
    cm.input["sources"] = [
        {
            "type": "planewave",
            "magnitudeFile": "gauss_1GHz.exc",
            "elementIds": [4],
            "direction": {"theta": 0.0, "phi": 0.0},
            "polarization": {"theta": 1.5708, "phi": 0.0}
        }
    ]

    # 6. Probes
    cm.input["probes"] = [
        {
            "name": "before",
            "type": "point",
            "field": "electric",
            "elementIds": [1],
            "directions": ["x"],
            "domain": {"type": "time"}
        },
        {
            "name": "inbox",
            "type": "point",
            "field": "electric",
            "elementIds": [2],
            "directions": ["x"],
            "domain": {"type": "time"}
        },
        {
            "name": "after",
            "type": "point",
            "field": "electric",
            "elementIds": [3],
            "directions": ["x"],
            "domain": {"type": "time"}
        }
    ]

    # 7. Save to file
    out_path = os.path.join(CASES_FOLDER, 'pw-in-box.fdtd.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(cm.input, f, indent=4)
        f.write('\n')

    print(f"✅ Arquivo 'pw-in-box.fdtd.json' salvo em '{out_path}' com sucesso.")

def create_excitation_pw():
    """
    Creates 'gauss_1GHz.exc' Gaussian pulse based on timeStep and numberOfSteps
    defined in 'pw-in-box.fdtd.json'. The file is saved in EXCITATIONS_FOLDER.
    """
    CASE_FILE = os.path.join(CASES_FOLDER,'pw-in-box.fdtd.json')

    # --- 1. Lê parâmetros do arquivo JSON
    if not os.path.isfile(CASE_FILE):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {CASE_FILE}")

    with open(CASE_FILE, 'r') as f:
        case_data = json.load(f)

    dt = case_data["general"]["timeStep"]
    n_steps = case_data["general"]["numberOfSteps"]

    # --- Parâmetros extraídos da curva do SEMBA
    dt = 1.8055e-13        # 0.18055 ps (intervalo médio do SEMBA)
    n_steps = 31073        # número de pontos
    t0 = 1.87e-9           # centro do pulso
    w0 = 0.187e-9          # largura do pulso

    # --- Sinal
    t = np.arange(n_steps) * dt
    src = np.exp(-((t - t0) / w0) ** 2)

    # --- 4. Salva o arquivo
    data = np.column_stack((t, src))
    output_path = os.path.join(EXCITATIONS_FOLDER, 'gauss_1GHz.exc')
    np.savetxt(output_path, data, fmt="%.5e", delimiter=' ')

    print(f"✅ Excitation file 'gauss_1GHz.exc' created successfully at {output_path}.")

def compare_excitations(file):
    """
    Plot a comparison between 'file1' and 'file2' located in the folder 'examplesData/excitations'.
    Assumes both files have two columns: time and amplitude.
    """
    ref_file = os.path.join(CWD_ROOT,'testData','cases','planewave', 'gauss_1GHz.exc')
    gen_file = os.path.join(CWD_ROOT,EXCITATIONS_FOLDER,file)

    # Verificação de existência
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"Arquivo de referência não encontrado: {ref_file}")
    if not os.path.isfile(gen_file):
        raise FileNotFoundError(f"Arquivo gerado não encontrado: {gen_file}")

    # Carregamento dos dados
    t_ref, e_ref = np.loadtxt(ref_file, unpack=True)
    t_gen, e_gen = np.loadtxt(gen_file, unpack=True)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(t_ref * 1e9, e_ref, label=ref_file, linestyle='--')
    plt.plot(t_gen * 1e9, e_gen, label=gen_file, linestyle='-')
    plt.xlabel('Time [ns]')
    plt.ylabel('Amplitude')
    plt.title('Comparação entre excitações gaussianas')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_simulation():
    """ Run the simulation and return the probe files. """

    # 1) Verifica a existência do semba-fdtd.exe e .fdtd.json
    if not os.path.isfile(SEMBA_EXE):
        raise FileNotFoundError(SEMBA_EXE)
    if not os.path.isfile(JSON_FILE):
        raise FileNotFoundError(JSON_FILE)

    # 2) Altera o diretório de trabalho para o diretório do caso
    # Isso é necessário para garantir que os arquivos de entrada
    # e saída sejam encontrados corretamente.
    os.chdir(CASES_FOLDER)

    try:
        # 3) Cria solver, limpa resíduos e executa a simulação
        solver = FDTD(input_filename=os.path.basename(JSON_FILE),
                      path_to_exe=SEMBA_EXE)
        solver.cleanUp()
        solver.run()

        # 4) Esse trecho busca os arquivos de saída das sondas before, inbox, after,
        # move cada um para a pasta de resultados outputs, e registra seus novos
        # caminhos num dicionário probes para facilitar o pós-processamento.

        # flowchart TD
        # A[Início: Cria dicionário vazio probes = {}] --> B
        # {Para cada name em ('before', 'inbox', 'after')}

        # B --> C[Chama solver.getSolvedProbeFilenames(name)]
        # C --> D[Pega o primeiro arquivo retornado (fn)]

        # D --> E[Constrói caminho origem src = CASES_FOLDER/fn]
        # E --> F[Constrói caminho destino dst = OUTPUTS_FOLDER/fn]

        # F --> G[Move arquivo de src para dst (os.replace)]

        # G --> H[Atualiza dicionário probes[name] = dst]

        # H --> I{Mais names?}
        # I -->|Sim| C
        # I -->|Não| J[Fim: dicionário probes completo]

        probes = {}  # (A)

        # (B) Para cada name em ('before', 'inbox', 'after')
        for name in ('before', 'inbox', 'after'):
            probe_files = solver.getSolvedProbeFilenames(name)  # (C)
            if not probe_files:
                print(f"⚠️ Atenção: Nenhum arquivo encontrado para probe '{name}'. Pulando.")
                continue

            fn = probe_files[0]  # (D)
            src = fn  # (E) já estamos em CASES_FOLDER
            dst = os.path.join(ABS_OUTPUTS_FOLDER, fn)  # (F)
            os.replace(src, dst)  # (G)

            # Mostra OUTPUTS_FOLDER relativo
            print(f"✅ Arquivo '{fn}' movido com sucesso para '{OUTPUTS_FOLDER}'.")

            probes[name] = dst  # (H)

        # 6) Move arquivos .txt para a pasta logs
        for file in os.listdir('.'):
            if file.endswith('.txt'):
                src_txt = os.path.abspath(file)  # caminho absoluto da origem
                dst_txt = os.path.join(LOGS_FOLDER, file)
                os.replace(src_txt, dst_txt)
                print(f"📄 Arquivo de log '{file}' movido com sucesso para '{LOGS_FOLDER}'.")

    finally:
        # 5) Volta para o diretório original
        os.chdir(CWD_ROOT)

    # 6) Retorna dicionário de probes
    return probes

def plot_probes(probes):
    """
    Plota cada probe em um subplot (1 linha x 3 colunas).
    """

    _, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 linha, 3 colunas

    # Ajusta a ordem esperada
    probe_order = ['before', 'inbox', 'after']

    for idx, name in enumerate(probe_order):
        if name in probes:
            probe = Probe(probes[name])
            ax = axes[idx]

            ax.plot(probe['time']*1e9, probe['field'], label=name)
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Field [V/m]')
            ax.set_title(f"Probe: {name}")
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Cria o caso de teste pw-in-box.fdtd.json
    # Caso o arquivo já exista, ele será sobrescrito.
    make_case_pw()

    # Cria o arquivo de excitação gauss_1GHz.exc
    # Caso o arquivo já exista, ele será sobrescrito.
    create_excitation_pw()

    # Compara os arquivos de excitação gerados
    compare_excitations('gauss_1GHz.exc')

    # Executa a simulação e obtém os arquivos de saída das sondas
    probe_list = run_simulation()
    plot_probes(probe_list)
    print("✅ Simulação concluída com sucesso!")
    print("✅ Probes:", probe_list)
