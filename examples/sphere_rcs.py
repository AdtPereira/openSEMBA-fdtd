r"""
sphere_rcs.py

Simulação FDTD para cálculo do RCS de uma esfera iluminada por uma onda plana.

------------------------------------------------------------
ESTE SCRIPT UTILIZA CAMINHOS ABSOLUTOS E NÃO ALTERA O DIRETÓRIO DE TRABALHO
------------------------------------------------------------

O script está estruturado para garantir que todos os arquivos e pastas usados
sejam acessados por caminhos absolutos, evitando problemas causados por mudanças
do diretório de trabalho (cwd).

PRINCIPAIS CONSIDERAÇÕES SOBRE DIRETÓRIOS:

1️⃣ Diretório raiz do projeto (CWD_ROOT):
    - CWD_ROOT = Path.cwd()
    - Definido automaticamente como o diretório atual quando o script é iniciado.
    - No seu ambiente, o diretório raiz é sempre:
      C:\\Users\\adilt\\OneDrive\\05_GIT\\openSEMBA\\fdtd
    - Todas as pastas e arquivos são referenciados em relação a esse diretório.

2️⃣ Estrutura esperada:
    - CWD_ROOT/
        ├── build/bin/semba-fdtd.exe
        ├── examples/
        ├── examplesData/
            ├── cases/
            ├── excitations/
            ├── logs/
            └── outputs/
        └── src_pyWrapper/

3️⃣ Caminhos absolutos:
    - Todas as variáveis de pastas (CASES_FOLDER, EXCITATIONS_FOLDER, LOGS_FOLDER, OUTPUTS_FOLDER)
      são definidas usando CWD_ROOT / caminho_relativo e convertidas em absolutos com .resolve().
    - Garante que os caminhos corretos sejam usados independentemente de onde o script é chamado.

4️⃣ Proibição do uso de os.chdir():
    - Não é usado os.chdir() no script.
    - Isso evita erros difíceis de rastrear quando o diretório de trabalho atual muda
      e faz com que caminhos relativos que antes funcionavam passem a falhar.

5️⃣ Uso do FDTD:
    - O arquivo JSON é passado com caminho absoluto para a classe FDTD.
    - Isso evita que o solver procure o JSON no diretório errado.

6️⃣ Organização das saídas:
    - Arquivos .txt e .pl são movidos para LOGS_FOLDER.
    - Arquivos .dat são movidos para OUTPUTS_FOLDER.
    - As operações de mover arquivos são feitas com verificação de existência
      para evitar erros se algum arquivo não for gerado.

IMPORTANTE:
-----------
Mesmo que o script seja executado a partir de diferentes pastas (ex: 'fdtd' ou 'fdtd/examples'),
os caminhos serão sempre resolvidos para os locais corretos.

------------------------------------------------------------
Execute sempre a partir do diretório raiz do projeto (fdtd):
C:\\Users\\adilt\\OneDrive\\05_GIT\\openSEMBA\\fdtd>
------------------------------------------------------------
"""
# pylint: disable=unused-import,wrong-import-position

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import FDTD, Probe

# Pastas principais
CASE_NAME   = 'SphereRCS'
SEMBA_EXE   = Path.cwd() / 'build' / 'bin' / 'semba-fdtd.exe'
EXAMPLES    = (Path.cwd() / 'examples').resolve()
CASES       = (Path.cwd() / 'examplesData' / 'cases').resolve()
EXCITATIONS = (Path.cwd() / 'examplesData' / 'excitations').resolve()
LOGS        = (Path.cwd() / 'examplesData' / 'logs').resolve()
OUTPUTS     = (Path.cwd() / 'examplesData' / 'outputs' / CASE_NAME).resolve()
JSON_FILE   = CASES / f'{CASE_NAME}.fdtd.json'

def ensure_folders_exist(*folders: list[Path]):
    """
    Cria os diretórios especificados se não existirem.
    Se o diretório já existir, não faz nada.
    """
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

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
        Nome do caso (por ex. 'SphereRCS'). Será usado para montar o nome
        do arquivo JSON e o caminho da subpasta ugrfdtd.
    examples_root : str, opcional
        Raiz dos exemplos (padrão apontando para ...\fdtd\examples).
    cases_root : str, opcional
        Diretório de destino onde os arquivos serão copiados
        (padrão ...\fdtd\examplesData\cases).
    """

    # Monta os caminhos de origem e destino
    src_dir = os.path.join(EXAMPLES, case_name, case_name, "ugrfdtd")

    files_to_copy = [
        "predefinedExcitation.1.exc",
        f"{case_name}.fdtd.json"
    ]

    for filename in files_to_copy:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(CASES, filename)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Arquivo de origem não encontrado: {src_path}")

        shutil.copy2(src_path, dst_path)
        print(f"✅ Arquivo '{filename}' copiado com sucesso para '{CASES}'")

def run_simulation():
    """Run the simulation and return o dicionário de probes com caminhos absolutos."""

    if not SEMBA_EXE.is_file():
        raise FileNotFoundError(SEMBA_EXE)
    if not JSON_FILE.is_file():
        raise FileNotFoundError(JSON_FILE)

    solver = FDTD(input_filename=str(JSON_FILE.resolve()),
              path_to_exe=str(SEMBA_EXE))
    solver.cleanUp()
    solver.run()
    if not solver.hasFinishedSuccessfully():
        raise RuntimeError("Solver não terminou com sucesso. Verifique o log.")

    probes = {}

    for fname in os.listdir(CASES):
        src = (CASES / fname).resolve()

        # 4) Move arquivos .txt e .pl para logs
        if fname.endswith('.txt') or fname.endswith('.pl'):
            dst = (LOGS / fname).resolve()
            if src.exists():
                os.replace(str(src), str(dst))
                print(f"📄 Arquivo de log '{fname}' movido para '{LOGS}'.")
            else:
                print(f"⚠ Aviso: '{fname}' não encontrado. Pulando.")

        # 5) Move arquivos .dat para OUTPUTS_FOLDER
        if fname.endswith('.dat'):
            dst = (OUTPUTS / fname).resolve()
            if src.exists():
                os.replace(str(src), str(dst))
                print(f"📊 Arquivo de saída '{fname}' movido para '{OUTPUTS}'.")
                probes[fname] = str(dst)
            else:
                print(f"⚠ Aviso: '{fname}' não encontrado. Pulando.")

        # 6) Move arquivos .exc para EXCITATIONS_FOLDER
        if fname.endswith('.exc'):
            dst = (EXCITATIONS / fname).resolve()
            if src.exists():
                os.replace(str(src), str(dst))
                print(f"📡 Arquivo de excitação '{fname}' movido para '{EXCITATIONS}'.")
            else:
                print(f"⚠ Aviso: '{fname}' não encontrado. Pulando.")

    return probes

def load_probes_from_outputs() -> dict[str, str]:
    """
    Varre `outputs_folder` em busca de todos os arquivos .dat de Point probes
    e retorna um dicionário {probe_filename: absolute_path}, compatível com
    o que run_simulation() retornaria.
    """
    probes = {}
    valid_tags = Probe.POINT_PROBE_TAGS  # ex: ['_Ex_', '_Ey_', '_Ez_', ...]

    for fname in os.listdir(OUTPUTS):
        if not fname.endswith('.dat'):
            continue
        if not any(tag in fname for tag in valid_tags):
            # pula outros .dat que não sejam sondas pontuais de E-field
            continue

        path = os.path.join(OUTPUTS, fname)
        try:
            p = Probe(path)
        except ValueError:
            # ignora arquivos com nome estranho
            continue

        if p.type == 'point' and p.field == 'E':
            probes[fname] = path

    return probes

def extract_point_probe_fields(probes: dict[str, str]) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Lê todos os probes do tipo 'point' com campo 'E' e agrupa os dados por nome e direção.

    Retorna
    -------
    dict:
        Estrutura { nome_da_sonda : { direção : DataFrame } }.
    """
    valid_tags = Probe.POINT_PROBE_TAGS
    point_groups = {}

    for filepath in probes.values():
        fname = os.path.basename(filepath)
        if not any(tag in fname for tag in valid_tags):
            continue
        try:
            p = Probe(filepath)
        except ValueError:
            continue

        # Força leitura confiável
        raw = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            skiprows=1,
            engine='python'
        )
        if raw.shape[1] == 3:
            raw.columns = ['time', 'field', 'incident']
        elif raw.shape[1] == 2:
            raw.columns = ['time', 'field']
        else:
            raw.columns = p.data.columns[: raw.shape[1]]
        p.data = raw

        if p.type == 'point' and p.field == 'E':
            point_groups.setdefault(p.name, {})[p.direction] = p.data

    return point_groups

def plot_point_probe_fields(point_groups: dict[str, dict[str, pd.DataFrame]]):
    """
    Plota os campos E total e incidente para cada direção de cada sonda.

    Parâmetros
    ----------
    point_groups : dict
        Estrutura { nome_da_sonda : { direção : DataFrame } }.
    """
    for probe_name, dirs in point_groups.items():
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        for idx, axis in enumerate(('x', 'y', 'z')):
            ax = axs[idx]
            if axis in dirs:
                df = dirs[axis]
                t_ns = df['time'] * 1e9

                ax.plot(t_ns, df['field'], label=f"E{axis.upper()} total")
                if 'incident' in df.columns:
                    ax.plot(
                        t_ns,
                        df['incident'],
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

def extract_far_field_data(field: str, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrai phi e o campo especificado (|Etheta|, |Ephi| ou RCS(GEOM)) 
    do arquivo Far Field para um valor de theta.

    Parâmetros
    ----------
    field : str
        Nome do campo. Deve ser: 'Etheta_mod', 'Ephi_mod' ou 'RCS(GEOM)'.
    theta_deg : float
        Ângulo θ em graus.

    Retorna
    -------
    phi : np.ndarray
        Ângulos phi [rad].
    values : np.ndarray
        Valores do campo selecionado.
    """
    prefix = f"{CASE_NAME}.fdtd_Far_FF"
    for fname in os.listdir(OUTPUTS.resolve()):
        if fname.startswith(prefix) and fname.endswith(".dat"):
            dat_path = os.path.join(OUTPUTS.resolve(), fname)
            break
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo começando com '{prefix}' em '{OUTPUTS.resolve()}'"
        )

    p = Probe(dat_path)
    df = p.data

    theta_rad = np.deg2rad(theta_deg)
    if "Theta" in df.columns:
        mask = np.isclose(df["Theta"].values, theta_rad)
    else:
        mask = np.isclose(df.iloc[:, 1].values, theta_rad)

    df_theta = df[mask].reset_index(drop=True)
    if df_theta.empty:
        raise ValueError(f"Nenhuma linha com Theta={theta_deg}° encontrada.")

    # Phi
    if "Phi" in df_theta.columns:
        phi = df_theta["Phi"].values
    else:
        phi = df_theta.iloc[:, 2].values

    # Campo
    if field in df_theta.columns:
        values = np.abs(df_theta[field].values)
    else:
        # Fallback: usa colunas por índice
        # (para compatibilidade com arquivos sem cabeçalho completo)
        index_map = {
            'Etheta_mod': 3,
            'Ephi_mod': 5,
            'RCS(GEOM)': 8
        }
        idx = index_map.get(field)
        if idx is None:
            raise ValueError(f"Campo {field} não suportado.")
        values = np.abs(df_theta.iloc[:, idx].values)

    return phi, values

def rcs_far_field_e_theta(theta_deg: float = 0.0):
    """
    Plota |Eθ| vs φ para um dado ângulo de observação θ (em graus),
    usando o arquivo Far Field gerado pelo FDTD cujo nome comece com
    CASE_NAME.fdtd_Far_FF*.dat em ABS_OUTPUTS_FOLDER.

    Parâmetros
    ----------
    theta_deg : float
        Ângulo de observação θ em graus (por ex. 0 ou 90). Default = 0.
    """
    phi, etheta = extract_far_field_data('Etheta_mod', theta_deg)

    _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(phi, etheta, "-", label=f"θ = {theta_deg}°")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ticks_deg = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])
    ax.set_title(rf"$|E_\theta|$ vs $\phi$ at $\theta={theta_deg}^\circ$")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()

def rcs_far_field_e_phi(theta_deg: float = 0.0):
    """
    Plota |Eφ| vs φ para um dado ângulo de observação θ (em graus),
    usando o arquivo Far Field gerado pelo FDTD cujo nome comece com
    CASE_NAME.fdtd_Far_FF*.dat em ABS_OUTPUTS_FOLDER.

    Parâmetros
    ----------
    theta_deg : float
        Ângulo de observação θ em graus (por ex. 0 ou 90). Default = 0.
    """
    phi, ephi = extract_far_field_data('Ephi_mod', theta_deg)

    _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(phi, ephi, "-", label=f"θ = {theta_deg}°")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ticks_deg = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])
    ax.set_title(rf"$|E_\phi|$ vs $\phi$ at $\theta={theta_deg}^\circ$")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()

def rcs_geom(theta_deg: float = 0.0):
    """
    Plota RCS_GEOM vs φ para um dado ângulo de observação θ (em graus),
    usando o arquivo Far Field gerado pelo FDTD cujo nome comece com
    CASE_NAME.fdtd_Far_FF*.dat em ABS_OUTPUTS_FOLDER.

    Parâmetros
    ----------
    theta_deg : float
        Ângulo de observação θ em graus (por ex. 0 ou 90). Default = 0.
    """
    phi, rcs = extract_far_field_data('RCS(GEOM)', theta_deg)

    _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(phi, rcs, "-", label=f"θ = {theta_deg}°")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ticks_deg = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])
    ax.set_title(rf"$\mathrm{{RCS}}_{{\rm GEOM}}$ vs $\phi$ at $\theta={theta_deg}^\circ$")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    # Garante a existência dos diretórios de saída e logs
    ensure_folders_exist(OUTPUTS, LOGS, EXCITATIONS)

    # Copia os arquivos de {case}\{case}\ugrfdtd para o diretório examplesData\cases
    # Executa a simulação e obtém os arquivos de saída das sondas
    copy_case_files(CASE_NAME)
    run_probes = run_simulation()

    # Carrega os arquivos de saída das sondas do diretório de saída
    # run_probes = load_probes_from_outputs()

    # Plota os resultados das sondas
    print(f"🔍 Probes encontrados: {list(run_probes.keys())}")
    point_data = extract_point_probe_fields(run_probes)
    plot_point_probe_fields(point_data)

    # Plota o RCS do far field
    # rcs_far_field_e_theta(theta_deg=90.0)
    # rcs_far_field_e_phi(theta_deg=90.0)
    # rcs_geom(theta_deg=90.0)
