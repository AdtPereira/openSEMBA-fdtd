r"""
cmd commands:
    conda activate semba-fdtd
    cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
    python examples/paul_8_6_square.py

Simulação FDTD para cálculo dos modos ressonantes de uma cavidade
esférica PEC excitada por um pulso gaussiano em um fio fino.

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
    - Todas as variáveis de pastas (CASES, EXCITATIONS, LOGS, OUTPUTS)
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
    - Arquivos .txt e .pl são movidos para LOGS.
    - Arquivos .dat são movidos para OUTPUTS.
    - Arquivos .exc são movidos para EXCITATIONS.
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
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import FDTD, Probe

# Pastas principais
CASE_NAME   = 'paul_8_6_square'
SEMBA_EXE   = Path.cwd() / 'build' / 'bin' / 'semba-fdtd.exe'
EXAMPLES    = (Path.cwd() / 'examples').resolve()
CASES       = (Path.cwd() / 'examplesData' / 'cases').resolve()
EXCITATIONS = (Path.cwd() / 'examplesData' / 'excitations').resolve()
LOGS        = (Path.cwd() / 'examplesData' / 'logs').resolve()
OUTPUTS     = (Path.cwd() / 'examplesData' / 'outputs' / CASE_NAME).resolve()
JSON_FILE   = CASES / f'{CASE_NAME}.fdtd.json'

def ensure_folders_exist(*folders: list[Path]) -> None:
    """
    Cria os diretórios especificados se não existirem.
    Se o diretório já existir, não faz nada.
    """
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

def create_excitation_file(
        exc_name: str = f"{CASE_NAME}.exc"
) -> None:
    """
    Cria um arquivo de excitação '{exc_name}' compatível com os parâmetros
    definidos no arquivo JSON do caso ('timeStep' e 'numberOfSteps').

    O arquivo de excitação é salvo na pasta CASES (mesma pasta do arquivo JSON).

    Justificativa:
    --------------
    O solver SEMBA-FDTD procura arquivos de excitação (.exc) no mesmo diretório
    onde o arquivo JSON (.fdtd.json) está localizado, **quando o JSON especifica
    apenas o nome do arquivo** no campo 'waveformFile' (sem caminho relativo).

    Exemplo no JSON:
    ----------------
    "waveformFile": "NodaVoltage.exc"

    Nesse caso, o solver busca:
    'examplesData/cases/NodaVoltage.exc'

    Se um caminho relativo completo fosse especificado no JSON, o arquivo
    precisaria ser salvo nessa pasta correspondente.

    Parâmetros
    ----------
    exc_name : str, opcional
        Nome do arquivo de excitação. Por padrão, usa '{CASE_NAME}.exc'.

    Detalhes do pulso:
    ------------------
    - Centro do pulso (t0): 0.5 * número de passos * dt
    - Largura do pulso (w0): 0.05 * número de passos * dt
    """

    # --- 1. Verifica se o JSON existe
    if not JSON_FILE.is_file():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {JSON_FILE}")

    # --- 2. Lê parâmetros do JSON
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        case_data = json.load(f)

    dt = case_data["general"]["timeStep"]
    n_steps = case_data["general"]["numberOfSteps"]

    # --- 3. Define o pulso de tensão
    # Pontos obtidos por interpolação linear da figura do artigo
    time_points = np.array(
        [0, 1.5, 3, 4.5, 6, 8, 10, 13, 16, 20, 25, 30, 35, 40]) * 1e-9
    voltage_points = np.array(
        [0, 7, 20, 35, 45, 50, 53, 55, 56, 57, 58, 58.5, 59, 60])

    # --- 4. Interpolação linear
    time = np.arange(n_steps) * dt
    voltage = np.interp(time, time_points, voltage_points)

    # --- 4. Salva o arquivo
    output_path = CASES / exc_name
    data = np.column_stack((time, voltage))
    np.savetxt(output_path, data, fmt="%.8e", delimiter=' ')

    print(f"✅ Arquivo de excitação '{exc_name}' criado com sucesso em: {output_path}")

def copy_json_file_from_ugrfdtd() -> None:
    r"""
    Copia apenas o arquivo JSON do caso:

      - '{CASE_NAME}.fdtd.json'

    Origem:
      {EXAMPLES}/{CASE_NAME}/{CASE_NAME}/ugrfdtd

    Destino:
      {CASES}

    Parâmetros
    ----------
    """
    src_dir = EXAMPLES / CASE_NAME / CASE_NAME / "ugrfdtd"
    json_filename = f"{CASE_NAME}.fdtd.json"
    src_path = src_dir / json_filename
    dst_path = CASES / json_filename

    if not src_path.is_file():
        raise FileNotFoundError(f"Arquivo JSON de origem não encontrado: {src_path}")

    shutil.copy2(src_path, dst_path)
    print(f"✅ Arquivo '{json_filename}' copiado com sucesso para '{CASES}'")

def copy_exc_file(
        exc_name: str = f"{CASE_NAME}.exc"
) -> None:
    r"""
    Copia o arquivo de excitação do caso:

      - '{CASE_NAME}.fdtd.json'

    Origem:
      {EXCITATIONS}

    Destino:
      {CASES}

    Parâmetros
    ----------
    """
    src_path = EXCITATIONS / exc_name
    dst_path = CASES / exc_name

    if not src_path.is_file():
        raise FileNotFoundError(f"Arquivo .exc de origem não encontrado: {src_path}")

    shutil.copy2(src_path, dst_path)
    print(f"✅ Arquivo '{exc_name}' copiado com sucesso para '{CASES}'")

def run_simulation() -> dict[str, str]:
    """
    Run a SEMBA-FDTD case end-to-end:

    1. Ensure OUTPUTS, LOGS and EXCITATIONS directories exist.
    2. Verify existence of SEMBA executable and input JSON.
    3. Read the JSON and stage any ".exc" files referenced under "sources" → "magnitudeFile":
       copy them from EXCITATIONS into CASES so the solver can find them.
    4. Invoke the FDTD solver; raise if it does not finish successfully.
    5. From CASES:
       • Move all ".txt" and ".pl" files to LOGS.
       • Move all ".dat" files to OUTPUTS, collecting their new paths into `probes`.
       • Delete any stray ".exc" files left in CASES (they stay in EXCITATIONS).
    6. Return a dict mapping each probe filename to its absolute OUTPUTS path.
    """
    # 1) Prepare folders
    ensure_folders_exist(OUTPUTS, LOGS, EXCITATIONS, CASES)

    # 2) Preconditions
    if not SEMBA_EXE.is_file():
        raise FileNotFoundError(f"SEMBA executable not found: {SEMBA_EXE}")
    if not JSON_FILE.is_file():
        raise FileNotFoundError(f"Input JSON not found: {JSON_FILE}")

    # 3) Stage excitation files from JSON → CASES
    with open(JSON_FILE, 'r', encoding='utf-8') as jf:
        cfg = json.load(jf)

    exc_names = [
        src.get("magnitudeFile")
        for src in cfg.get("sources", [])
        if src.get("magnitudeFile")
    ]
    for name in exc_names:
        src = EXCITATIONS / name
        dst = CASES      / name
        if src.is_file():
            shutil.copy2(str(src), str(dst))
            print(f"📡 Copied excitation '{name}' → CASES (original kept in EXCITATIONS)")
        else:
            print(f"⚠ Excitation '{name}' not found in EXCITATIONS; skipping")

    # 4) Run the solver
    solver = FDTD(
        input_filename=str(JSON_FILE.resolve()),
        path_to_exe=str(SEMBA_EXE.resolve())
    )
    solver.cleanUp()
    solver.run()
    if not solver.hasFinishedSuccessfully():
        raise RuntimeError("Solver did not finish successfully; check logs.")

    # 5) Harvest outputs and clean up CASES
    probes: dict[str, str] = {}
    for path in CASES.iterdir():
        if path.suffix in ('.txt', '.pl'):
            dest = LOGS / path.name
            path.replace(dest)
            print(f"📄 Log '{path.name}' → LOGS")
        elif path.suffix == '.dat':
            dest = OUTPUTS / path.name
            path.replace(dest)
            probes[path.name] = str(dest.resolve())
            print(f"📊 Output '{path.name}' → OUTPUTS")
        elif path.suffix == '.exc':
            path.unlink()
            print(f"🗑️ Removed stray excitation '{path.name}' from CASES")

    return probes

def plot_excitation_file(
        exc_name: str = f"{CASE_NAME}.exc"
) -> None:
    """
    Plota a curva de tensão do arquivo .exc.

    Parâmetros
    ----------
    exc_name : str, opcional

    O arquivo será lido da pasta EXCITATIONS (onde o run_simulation()
    move o arquivo após a simulação).
    """
    exc_path = EXCITATIONS / exc_name

    if not exc_path.is_file():
        raise FileNotFoundError(f"Arquivo de excitação não encontrado: {exc_path}")

    # Carrega os dados
    data = np.loadtxt(exc_path)
    time = data[:, 0] * 1e9  # converter para ns
    voltage = data[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(time, voltage, label=f'{exc_name}', color='blue')
    plt.xlabel("Tempo [ns]")
    plt.ylabel("Tensão [V]")
    plt.xlim(0, time[-1])
    plt.title("Curva de excitação de tensão")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

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

def extract_point_probe_scalars(
        probes: dict[str, str]
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Lê todos os probes multiconductor (_V_ e _I_) e agrupa cada DataFrame
    por nome de sonda nas chaves 'voltage' e 'current'.

    Retorna
    -------
    dict:
        { probe_name: {
            'voltage': DataFrame(time, value),    # ou ausente se não houver
            'current': DataFrame(time, value)     # ou ausente se não houver
          }
        }
    """
    scalar_groups: dict[str, dict[str, pd.DataFrame]] = {}

    for path in probes.values():
        try:
            p = Probe(path)
        except Exception:
            continue

        # filtra só as sondas MTLN (_V_, _I_) – não há p.field nem p.direction nesse ramo
        if p.type != 'mtln':
            continue

        df = p.data
        grp = scalar_groups.setdefault(p.name, {})

        # procura colunas voltage_* e current_*
        volt_cols = [c for c in df.columns if c.startswith('voltage')]
        curr_cols = [c for c in df.columns if c.startswith('current')]

        # guarda o primeiro voltage_* se existir
        if volt_cols:
            # renomeia para ['time','value']
            v = df[['time', volt_cols[0]]].rename(
                columns={volt_cols[0]: 'value'}
            )
            grp['voltage'] = v

        # guarda o primeiro current_* se existir
        if curr_cols:
            i = df[['time', curr_cols[0]]].rename(
                columns={curr_cols[0]: 'value'}
            )
            grp['current'] = i

    return scalar_groups

def plot_point_probe_scalars(
        scalar_groups: dict[str, dict[str, pd.DataFrame]],
        time_unit: str = 's'
) -> None:
    """
    Plot voltage and current time‐series extracted from point probes.

    Parameters
    ----------
    scalar_groups : dict[str, dict[str, pd.DataFrame]]
        Output of `extract_point_probe_scalars`, i.e.
        { probe_name: { 'voltage': DataFrame, 'current': DataFrame } }.
        Each DataFrame must have columns ['time', 'value', ...].
    time_unit : str, optional
        Text label for the time axis (e.g. 's', 'ns'), by default 's'.

    Behavior
    --------
    For each probe in `scalar_groups`, creates a new figure with:
    - Voltage vs time (if present)
    - Current vs time (if present)
    """
    for probe_name, signals in scalar_groups.items():
        plt.figure(figsize=(8, 4))

        # plot voltage
        if 'voltage' in signals:
            df_v = signals['voltage']
            plt.plot(df_v['time'], df_v['value'], label='Voltage')

        # plot current
        if 'current' in signals:
            df_i = signals['current']
            plt.plot(df_i['time'], df_i['value'], label='Current')

        plt.title(f'Probe: {probe_name}')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Cria o arquivo de excitação com base no JSON
    # create_excitation_file()
    
    # Plota a curva de excitação
    plot_excitation_file(exc_name="coaxial_line_paul_8_6_0.25_square.exc")

    # Executa a simulação e obtém os arquivos de saída das sondas
    run_probes = run_simulation()
    print(f"🔍 Probes encontrados: {list(run_probes.keys())}")

    # Carrega os arquivos de saída das sondas do diretório de saída
    # run_probes = load_probes_from_outputs()
    # print(f"🔍 Probes encontrados: {list(run_probes.keys())}")

    # Extrai os dados escalares de tensão e corrente
    scalar_groups = extract_point_probe_scalars(run_probes)
    print(f"🔍 Grupos encontrados: {list(scalar_groups.keys())}")

    # Plota os dados escalares de tensão e corrente
    plot_point_probe_scalars(scalar_groups, time_unit='ns')
    
    print("✅ Simulação concluída com sucesso.")