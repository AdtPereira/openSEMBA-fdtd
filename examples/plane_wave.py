'''
Plane wave in a box example.
This example shows how to run a simulation of a plane wave in a box using the openSEMBA FDTD solver.

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

# conda activate semba-fdtd
# cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
# python examples/plane_wave.py

import os
import sys
import matplotlib.pyplot as plt

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import FDTD, Probe # pylint: disable=unused-import,wrong-import-position

# Define o caminho absoluto para o executável semba-fdtd.exe
# Diretório onde estão JSON e .exc
SEMBA_EXE           = os.path.abspath(os.path.join('build','bin','semba-fdtd.exe'))
CASES_FOLDER        = os.path.join('examplesData','cases')
JSON_FILE           = os.path.join(CASES_FOLDER, 'pw-in-box.fdtd.json')
OUTPUTS_FOLDER      = os.path.join('examplesData','outputs', 'plane_wave')
ABS_OUTPUTS_FOLDER  = os.path.abspath(OUTPUTS_FOLDER)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

def run_simulation():
    """ Run the simulation and return the probe files. """

    # 1) Verifica a existência do semba-fdtd.exe e .fdtd.json
    if not os.path.isfile(SEMBA_EXE):
        raise FileNotFoundError(SEMBA_EXE)
    if not os.path.isfile(JSON_FILE):
        raise FileNotFoundError(JSON_FILE)

    # 2) Salva diretório atual e muda para CASES_FOLDER
    cwd_root = os.getcwd()
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
        # A[Início: Cria dicionário vazio probes = {}] --> B{Para cada name em ('before', 'inbox', 'after')}

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

            src = fn  # já estamos em CASES_FOLDER
            dst = os.path.join(ABS_OUTPUTS_FOLDER, fn)  # (F)

            os.makedirs(ABS_OUTPUTS_FOLDER, exist_ok=True)
            os.replace(src, dst)  # (G)
            print(f"✅ Arquivo '{fn}' movido com sucesso para '{OUTPUTS_FOLDER}'.")  # Mostra OUTPUTS_FOLDER relativo

            probes[name] = dst  # (H)

        # (I) Se houver mais names, repete; senão, finaliza

    finally:
        # 5) Volta para o diretório original
        os.chdir(cwd_root)

    # 6) Retorna dicionário de probes
    return probes 

def plot_probes(probes):
    """
    Plota cada probe em um subplot (1 linha x 3 colunas).
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 linha, 3 colunas

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
    probes = run_simulation()
    plot_probes(probes)
    print("✅ Simulação concluída com sucesso!")
    print("✅ Probes:", probes)