r"""
    Plane wave scattering from a cylinder in a box using openSEMBA FDTD solver.
    This example shows how to run a simulation the openSEMBA FDTD solver 
    from examples\cylinder_rcs\cylinder_rcs\ugrfdtd\cylinder_rcs.fdtd.json.

        cmd commands:
                conda activate semba-fdtd

        cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
        python examples/cylinder_rcs.py

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
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import * # pylint: disable=unused-import,wrong-import-position

# Define o diretório atual (cwd) como o diretório raiz do projeto
# CWD_ROOT = C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
CWD_ROOT = os.getcwd()
CASE_NAME = 'cylinder_rcs'

# Define o caminho absoluto para o executável semba-fdtd.exe
SEMBA_EXE           = os.path.abspath(os.path.join('build','bin','semba-fdtd.exe'))
EXAMPLES_FOLDER      = os.path.join(CWD_ROOT,'examples')
CASES_FOLDER        = os.path.join('examplesData','cases')
EXCITATIONS_FOLDER  = os.path.join('examplesData','excitations')
LOGS_FOLDER         = os.path.join(CWD_ROOT,'examplesData','logs')

# Define o caminho absoluto para o diretório de saída dos arquivos .dat
JSON_FILE           = os.path.join(CASES_FOLDER,f'{CASE_NAME}.fdtd.json')
OUTPUTS_FOLDER      = os.path.join('examplesData','outputs', CASE_NAME)
ABS_OUTPUTS_FOLDER  = os.path.abspath(OUTPUTS_FOLDER) # "...\examplesData\outputs\cylinder_rcs"

# Cria os diretórios de saída e logs, se não existirem
os.makedirs(ABS_OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(EXCITATIONS_FOLDER, exist_ok=True)

def copy_case_files(case_name: str):
    r"""
    Copia os arquivos:
      - 'predefinedExcitation.1.exc'
      - '{case_name}.fdtd.json'
    do diretório:
      examples_root\{case_name}\{case_name}\ugrfdtd
    para o diretório:
      cases_root

    Parâmetros
    ----------
    case_name : str
        Nome do caso (por ex. 'cylinder_rcs'). Será usado para montar o nome
        do arquivo JSON e o caminho da subpasta ugrfdtd.
    examples_root : str, opcional
        Raiz dos exemplos (padrão apontando para ...\fdtd\examples).
    cases_root : str, opcional
        Diretório de destino onde os arquivos serão copiados
        (padrão ...\fdtd\examplesData\cases).
    """

    # Monta os caminhos de origem e destino
    src_dir = os.path.join(EXAMPLES_FOLDER, case_name, case_name, "ugrfdtd")

    files_to_copy = [
        "predefinedExcitation.1.exc",
        f"{case_name}.fdtd.json"
    ]

    for filename in files_to_copy:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(CASES_FOLDER, filename)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Arquivo de origem não encontrado: {src_path}")

        shutil.copy2(src_path, dst_path)
        print(f"✅ Arquivo '{filename}' copiado com sucesso para '{CASES_FOLDER}'")

def run_simulation():
    """ Run the simulation and return o dicionário de probes com caminhos absolutos. """

    # 1) Verifica a existência do semba-fdtd.exe e do JSON de entrada
    if not os.path.isfile(SEMBA_EXE):
        raise FileNotFoundError(SEMBA_EXE)
    if not os.path.isfile(JSON_FILE):
        raise FileNotFoundError(JSON_FILE)

    # 2) Vai para o diretório do caso
    os.chdir(CASES_FOLDER)

    try:
        # 3) Executa solver
        solver = FDTD(input_filename=os.path.basename(JSON_FILE),
                      path_to_exe=SEMBA_EXE)
        solver.cleanUp()
        solver.run()

        probes = {}

        # 4) Move arquivos .txt e .pl para logs
        for fname in os.listdir('.'):
            if fname.endswith('.txt') or fname.endswith('.pl'):
                src = os.path.abspath(fname)
                dst = os.path.join(LOGS_FOLDER, fname)
                os.replace(src, dst)
                print(f"📄 Arquivo de log '{fname}' movido para '{LOGS_FOLDER}'.")

        # 5) Move arquivos .dat para OUTPUTS_FOLDER
        for fname in os.listdir('.'):
            if fname.endswith('.dat'):
                src = os.path.abspath(fname)
                dst = os.path.join(ABS_OUTPUTS_FOLDER, fname)
                shutil.move(src, dst)
                print(f"📊 Arquivo de saída '{fname}' movido para '{ABS_OUTPUTS_FOLDER}'.")
                # opcional: registre no dicionário de probes, se precisar usar depois
                probes[fname] = dst

    finally:
        # 6) Sempre volte para o diretório original
        os.chdir(CWD_ROOT)

    return probes

def load_probes_from_outputs(outputs_folder: str) -> dict:
    """
    Varre `outputs_folder` em busca de todos os arquivos .dat de Point probes
    e retorna um dicionário {probe_filename: absolute_path}, compatível com
    o que run_simulation() retornaria.
    """
    probes = {}
    valid_tags = Probe.POINT_PROBE_TAGS  # ex: ['_Ex_', '_Ey_', '_Ez_', ...]

    for fname in os.listdir(outputs_folder):
        if not fname.endswith('.dat'):
            continue
        if not any(tag in fname for tag in valid_tags):
            continue # pula outros .dat que não sejam sondas pontuais de E-field

        path = os.path.join(outputs_folder, fname)
        try:
            p = Probe(path)
        except ValueError:
            continue # ignora arquivos com nome estranho

        if p.type == 'point' and p.field == 'E':
            probes[fname] = path

    return probes

def plot_point_probe_fields(probes: dict):
    """
    Plota Ex, Ey e Ez (total e incidente) para cada Point probe em `probes`.
    Se o .dat tiver três colunas, interpreta sempre a 3ª como incidente, 
    independente do cabeçalho.
    """
    valid_tags = Probe.POINT_PROBE_TAGS
    point_groups = {}

    # 1) Leia cada .dat e agrupe somente os Point probes de E-field
    for filepath in probes.values():
        fname = os.path.basename(filepath)
        if not any(tag in fname for tag in valid_tags):
            continue
        try:
            p = Probe(filepath)
        except ValueError:
            continue

        # (Re)leitura simplificada: ignora cabeçalho e força sep por regex
        raw = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            skiprows=1,
            engine='python'
        )
        # atribui nomes conforme nº de colunas
        if raw.shape[1] == 3:
            raw.columns = ['time', 'field', 'incident']
        elif raw.shape[1] == 2:
            raw.columns = ['time', 'field']
        else:
            # fallback nos primeiros rótulos do Probe original
            raw.columns = p.data.columns[: raw.shape[1]]
        p.data = raw

        if p.type == 'point' and p.field == 'E':
            point_groups.setdefault(p.name, {})[p.direction] = p

    # 2) Plot em subplots 3×1 para cada probe
    for probe_name, dirs in point_groups.items():
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        for idx, axis in enumerate(('x', 'y', 'z')):
            ax = axs[idx]
            if axis in dirs:
                p = dirs[axis]
                t_ns = p.data['time'] * 1e9

                ax.plot(t_ns, p.data['field'], label=f"E{axis.upper()} total")
                if 'incident' in p.data.columns:
                    ax.plot(
                        t_ns,
                        p.data['incident'],
                        label=f"E{axis.upper()} incidente",
                        linestyle='--'
                    )

                ax.set_ylabel(f"E{axis.upper()} [V/m]")
                ax.legend()
                ax.grid(True)
            else:
                ax.text(
                    0.5, 0.5,
                    f"Sem E{axis.upper()}",
                    ha='center', va='center'
                )
                ax.set_ylabel(f"E{axis.upper()}")

        axs[-1].set_xlabel("Tempo [ns]")
        fig.suptitle(f"Point Probe: {probe_name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_rcs_far_field_e_theta(theta_deg: float = 0.0):
    """
    Plota |Eθ| vs φ para um dado ângulo de observação θ (em graus),
    usando o arquivo Far Field gerado pelo FDTD cujo nome comece com
    CASE_NAME.fdtd_Far_FF*.dat em ABS_OUTPUTS_FOLDER.

    Parâmetros
    ----------
    theta_deg : float
        Ângulo de observação θ em graus (por ex. 0 ou 90). Default = 0.
    """
    # 1) Encontre o .dat de Far Field
    prefix = f"{CASE_NAME}.fdtd_Far_FF"
    for fname in os.listdir(ABS_OUTPUTS_FOLDER):
        if fname.startswith(prefix) and fname.endswith(".dat"):
            dat_path = os.path.join(ABS_OUTPUTS_FOLDER, fname)
            break
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo começando com '{prefix}' em '{ABS_OUTPUTS_FOLDER}'"
        )

    # 2) Carrega com Probe
    p = Probe(dat_path)
    df = p.data

    # 3) Converte theta para radianos e filtra
    theta_rad = np.deg2rad(theta_deg)
    if "Theta" in df.columns:
        mask = np.isclose(df["Theta"].values, theta_rad)
    else:
        mask = np.isclose(df.iloc[:, 1].values, theta_rad)
    df_theta = df[mask].reset_index(drop=True)
    if df_theta.empty:
        raise ValueError(f"Nenhuma linha com Theta={theta_deg}° encontrada.")

    # 4) Extrai phi (já em radianos no arquivo) e |Etheta_mod|
    if "Phi" in df_theta.columns:
        phi = df_theta["Phi"].values
    else:
        phi = df_theta.iloc[:, 2].values

    if "Etheta_mod" in df_theta.columns:
        etheta = np.abs(df_theta["Etheta_mod"].values)
    else:
        etheta = np.abs(df_theta.iloc[:, 3].values)

    # 5) Plota em projeção polar, mas com ticks em graus
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(phi, etheta, "-", label=f"θ = {theta_deg}°")

    ax.set_theta_zero_location("E")   # 0° à direita
    ax.set_theta_direction(-1)        # ângulos crescem horário

    # Ticks de ângulo em graus
    ticks_deg = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])

    ax.set_title(rf"$|E_\theta|$ vs $\phi$ at $\theta={theta_deg}^\circ$")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()

def plot_rcs_far_field_e_phi(theta_deg: float = 0.0):
    """
    Plota |Eφ| vs φ para um dado ângulo de observação θ (em graus),
    usando o arquivo Far Field gerado pelo FDTD cujo nome comece com
    CASE_NAME.fdtd_Far_FF*.dat em ABS_OUTPUTS_FOLDER.

    Parâmetros
    ----------
    theta_deg : float
        Ângulo de observação θ em graus (por ex. 0 ou 90). Default = 0.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from src_pyWrapper.pyWrapper import Probe

    # 1) Encontre o .dat de Far Field
    prefix = f"{CASE_NAME}.fdtd_Far_FF"
    for fname in os.listdir(ABS_OUTPUTS_FOLDER):
        if fname.startswith(prefix) and fname.endswith(".dat"):
            dat_path = os.path.join(ABS_OUTPUTS_FOLDER, fname)
            break
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo começando com '{prefix}' em '{ABS_OUTPUTS_FOLDER}'"
        )

    # 2) Carrega com Probe
    p = Probe(dat_path)
    df = p.data

    # 3) Converte theta para radianos e filtra
    theta_rad = np.deg2rad(theta_deg)
    if "Theta" in df.columns:
        mask = np.isclose(df["Theta"].values, theta_rad)
    else:
        mask = np.isclose(df.iloc[:, 1].values, theta_rad)
    df_theta = df[mask].reset_index(drop=True)
    if df_theta.empty:
        raise ValueError(f"Nenhuma linha com Theta={theta_deg}° encontrada.")

    # 4) Extrai phi (já em radianos no arquivo) e |Ephi_mod|
    if "Phi" in df_theta.columns:
        phi = df_theta["Phi"].values
    else:
        phi = df_theta.iloc[:, 2].values

    if "Ephi_mod" in df_theta.columns:
        ephi = np.abs(df_theta["Ephi_mod"].values)
    else:
        # fallback para a coluna de módulo Eφ por índice
        ephi = np.abs(df_theta.iloc[:, 5].values)

    # 5) Plota em projeção polar, mas com ticks em graus
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(phi, ephi, "-", label=f"θ = {theta_deg}°")

    ax.set_theta_zero_location("E")   # 0° à direita
    ax.set_theta_direction(-1)        # ângulos crescem no sentido horário

    # Ticks de ângulo em graus
    ticks_deg = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])

    ax.set_title(rf"$|E_\phi|$ vs $\phi$ at $\theta={theta_deg}^\circ$")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()

def plot_rcs_geom(theta_deg: float = 0.0):
    """
    Plota RCS_GEOM vs φ para um dado ângulo de observação θ (em graus),
    usando o arquivo Far Field gerado pelo FDTD cujo nome comece com
    CASE_NAME.fdtd_Far_FF*.dat em ABS_OUTPUTS_FOLDER.

    Parâmetros
    ----------
    theta_deg : float
        Ângulo de observação θ em graus (por ex. 0 ou 90). Default = 0.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from src_pyWrapper.pyWrapper import Probe

    # 1) Encontre o arquivo .dat de Far Field
    prefix = f"{CASE_NAME}.fdtd_Far_FF"
    for fname in os.listdir(ABS_OUTPUTS_FOLDER):
        if fname.startswith(prefix) and fname.endswith(".dat"):
            dat_path = os.path.join(ABS_OUTPUTS_FOLDER, fname)
            break
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo começando com '{prefix}' em '{ABS_OUTPUTS_FOLDER}'"
        )

    # 2) Carrega com Probe
    p = Probe(dat_path)
    df = p.data

    # 3) Converte theta para radianos e filtra
    theta_rad = np.deg2rad(theta_deg)
    if "Theta" in df.columns:
        mask = np.isclose(df["Theta"].values, theta_rad)
    else:
        mask = np.isclose(df.iloc[:, 1].values, theta_rad)
    df_theta = df[mask].reset_index(drop=True)
    if df_theta.empty:
        raise ValueError(f"Nenhuma linha com Theta={theta_deg}° encontrada em {fname}")

    # 4) Extrai φ (já em radianos) e RCS_GEOM
    if "Phi" in df_theta.columns:
        phi = df_theta["Phi"].values
    else:
        phi = df_theta.iloc[:, 2].values

    if "RCS(GEOM)" in df_theta.columns:
        rcs = df_theta["RCS(GEOM)"].values
    else:
        rcs = df_theta.iloc[:, 8].values

    # 5) Plota em projeção polar com ticks em graus
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(phi, rcs, "-", label=f"θ = {theta_deg}°")

    ax.set_theta_zero_location("E")   # 0° à direita
    ax.set_theta_direction(-1)        # ângulos crescem horário

    # Ticks de ângulo em graus
    ticks_deg = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])

    ax.set_title(rf"$\mathrm{{RCS}}_{{\rm GEOM}}$ vs $\phi$ at $\theta={theta_deg}^\circ$")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    # Copia os arquivos do caso {case}\{case}\ugrfdtd para o diretório examplesData\cases
    # copy_case_files(CASE_NAME)

    # Executa a simulação e obtém os arquivos de saída das sondas
    # probes = run_simulation()

    # Plota os resultados das sondas
    probes = load_probes_from_outputs(ABS_OUTPUTS_FOLDER)
    print(f"🔍 Probes encontrados: {list(probes.keys())}")
    plot_point_probe_fields(probes)

    # # Plota o RCS do far field
    # # plot_rcs_far_field_e_theta(theta_deg=90.0)
    # # plot_rcs_far_field_e_phi(theta_deg=90.0)
    # plot_rcs_geom(theta_deg=90.0)