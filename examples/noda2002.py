r"""
noda2002.py

Simulação FDTD para cálculo da impedância de surto de uma configuração
de horizontal de condutor.

Baseado no exemplo V.A de Noda (2002):
T. Noda and S. Yokoyama, "Thin wire representation in finite difference
time domain surge simulation," in IEEE Transactions on Power Delivery,
vol. 17, no. 3, pp. 840-847, July 2002, doi: 10.1109/TPWRD.2002.1022813.

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
cmd commands:
    conda activate semba-fdtd
    cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
    python examples/noda2002.py
"""
# pylint: disable=unused-import,wrong-import-position

import os
import sys
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Dict

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import FDTD, Probe

# Pastas principais
CASE_NAME   = 'noda2002'
SEMBA_EXE   = Path.cwd() / 'build' / 'bin' / 'semba-fdtd.exe'
EXAMPLES    = (Path.cwd() / 'examples').resolve()
CASES       = (Path.cwd() / 'examplesData' / 'cases').resolve()
EXCITATIONS = (Path.cwd() / 'examplesData' / 'excitations').resolve()
LOGS        = (Path.cwd() / 'examplesData' / 'logs').resolve()
OUTPUTS     = (Path.cwd() / 'examplesData' / 'outputs' / CASE_NAME).resolve()
INPUTS      = (Path.cwd() / 'examplesData' / 'inputs' / CASE_NAME).resolve()
JSON_FILE   = CASES / f'{CASE_NAME}.fdtd.json'

def ensure_folders_exist(*folders: list[Path]) -> None:
    """
    Cria os diretórios especificados se não existirem.
    Se o diretório já existir, não faz nada.
    """
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

def create_excitation_file(exc_name: str = f"{CASE_NAME}.exc") -> None:
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

def run_simulation() -> dict[str, str]:
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

def extract_point_probe_data(probes: dict[str, str]) -> dict[str, dict[str, pd.DataFrame]]:
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

def plot_point_probe_data(point_groups: dict[str, dict[str, pd.DataFrame]]) -> None:
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

def extract_probe_wire_data(
    outputs_folder: str = OUTPUTS
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Varre a pasta de saídas em busca de todos os arquivos de corrente de fio
    (*.fdtd_Wire probe_*.dat), carrega todas as colunas de cada um e retorna
    um dict onde as chaves são os nomes de arquivo e os valores são, por sua vez,
    dicts com arrays numpy para cada coluna.

    Parameters
    ----------
    outputs_folder : str
        Caminho para a pasta onde estão os .dat (por padrão, OUTPUTS_FOLDER).

    Returns
    -------
    all_data : Dict[str, Dict[str, np.ndarray]]
        Exemplo de estrutura de retorno:
        {
            'noda2002.fdtd_Wire probe_Wz_12_12_3_s2.dat': {
                'time'          : array([...]),
                'current'       : array([...]),
                'E_dl'          : array([...]),
                'Vplus'         : array([...]),
                'Vminus'        : array([...]),
                'Vplus-Vminus'  : array([...]),
            },
            'outroCaso.fdtd_Wire probe_Wy_...' : { … },
            ...
        }
    """
    all_data: Dict[str, Dict[str, np.ndarray]] = {}

    for fname in os.listdir(outputs_folder):
        if fname.endswith('.dat') and 'fdtd_Wire_probe_' in fname:
            path = os.path.join(outputs_folder, fname)
            # carrega tudo, pulando o header de uma linha
            arr = np.loadtxt(path, skiprows=1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            all_data[fname] = {
                'time'          : arr[:, 0],
                'current'       : arr[:, 1],
                'E_dl'          : arr[:, 2],
                'Vplus'         : arr[:, 3],
                'Vminus'        : arr[:, 4],
                'Vplus-Vminus'  : arr[:, 5],
            }

    return all_data

def plot_probe_wire_data(
    field: str,
) -> None:
    """
    Plota o campo especificado ('current', 'E_dl', 'Vplus', 'Vminus', ou 'Vplus-Vminus')
    em função do tempo para todos os arquivos de corrente de fio encontrados
    em OUTPUTS.

    Parâmetros
    ----------
    field : str
        Nome do campo a ser plotado. Deve ser uma das chaves:
        'current', 'E_dl', 'Vplus', 'Vminus', 'Vplus-Vminus'.

    Exemplo de uso
    -------------
    >>> plot_point_probe_field('current')
    >>> plot_point_probe_field('E_dl')
    """
    data_dict = extract_probe_wire_data(OUTPUTS)
    if not data_dict:
        raise RuntimeError(f"Nenhum arquivo de corrente encontrado em '{OUTPUTS}'")

    # verifica se o campo existe em pelo menos um arquivo
    sample = next(iter(data_dict.values()))
    if field not in sample:
        raise ValueError(f"Campo '{field}' inválido. Escolha entre {list(sample.keys())}")

    plt.figure(figsize=(8, 5))
    for fname, d in data_dict.items():
        plt.plot(d['time'], d[field], label=fname.replace('.dat', ''))

    plt.xlabel('Tempo (s)')
    ylabel_map = {
        'current': 'Corrente (A)',
        'E_dl'   : r'$\int E \cdot dl$ (V)',
    }
    plt.ylabel(ylabel_map.get(field, field))
    plt.title(f"{ylabel_map.get(field, field)} vs Tempo")
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_excitation_vs_Edl(
    exc_name: str = f"{CASE_NAME}.exc",
) -> None:
    """
    Compara a curva de excitação (.exc) com a coluna E_dl de todas as sondas
    de fio encontradas em `outputs_folder`.

    Parâmetros
    ----------
    excitation_path : Path
        Caminho para o arquivo .exc contendo duas colunas (tempo, amplitude).
    outputs_folder : Path
        Pasta onde estão os .dat das sondas de corrente (padrão: OUTPUTS).
    skiprows_exc : int
        Quantas linhas de cabeçalho pular ao ler o .exc (padrão: 1).
    """
    # 1) Carrega excitação
    exc_path = EXCITATIONS / exc_name
    if not exc_path.is_file():
        raise FileNotFoundError(f"Arquivo de excitação não encontrado: {exc_path}")

    # Carrega os dados
    data = np.loadtxt(exc_path)
    exc_time = data[:, 0] * 1e9  # converter para ns
    exc_voltage = data[:, 1]

    # 2) Carrega E_dl de cada sonda
    all_data: Dict[str, Dict[str, np.ndarray]] = extract_probe_wire_data(OUTPUTS)
    if not all_data:
        raise RuntimeError(f"Nenhum arquivo de fio encontrado em '{OUTPUTS}'")

    # 3) Plota tudo em um único gráfico
    plt.figure(figsize=(8, 5))
    # Curva de excitação (pontilhada)
    plt.plot(exc_time, exc_voltage, '--', label=EXCITATIONS.name)

    # Curvas E_dl de cada probe
    for fname, cols in all_data.items():
        plt.plot(cols['time'] * 1e9, cols['E_dl'], label=fname.replace('.dat', ''))
    plt.xlabel("Tempo [ns]")
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def surge_impedance(
        exc_name: str = f"{CASE_NAME}.exc", t_in_ns: float = 15.0, t_fi_ns: float = 20.0
        ) -> float:
    """
    Calcula a impedância de surto como a razão entre as médias
    de V e I no intervalo de tempo especificado (padrão 15-20 ns).

    Parâmetros
    ----------
    exc_name : str, opcional
        Nome do arquivo de excitação. Por padrão, 'NodaVoltage.exc'.

    t_in_ns : float
        Tempo inicial do intervalo [ns].

    t_fi_ns : float
        Tempo final do intervalo [ns].

    Retorna
    -------
    Zs : float
        Impedância de surto estimada [Ohms].
    """

    # --- 1. Carrega a tensão
    exc_path = EXCITATIONS / exc_name
    if not exc_path.is_file():
        raise FileNotFoundError(f"Arquivo de excitação não encontrado: {exc_path}")
    exc_data = np.loadtxt(exc_path)
    t_v = exc_data[:, 0] * 1e9  # tempo em ns
    v = exc_data[:, 1]

    # --- 2. Carrega a corrente (Wire probe)
    wire_file = next(OUTPUTS.glob(f"{CASE_NAME}.fdtd_Wire probe_Wz_*.dat"))
    current_data = np.loadtxt(wire_file, skiprows=1)
    t_i = current_data[:, 0] * 1e9  # tempo em ns
    i = current_data[:, 1]

    # --- 3. Interpolação (caso os tempos não coincidam exatamente)
    i_interp = interp1d(t_i, i, kind='linear', bounds_error=False, fill_value="extrapolate")
    i_resampled = i_interp(t_v)

    # --- 4. Seleção do intervalo de tempo
    mask = (t_v >= t_in_ns) & (t_v <= t_fi_ns)
    if not np.any(mask):
        raise ValueError("Nenhum dado encontrado no intervalo especificado.")

    v_avg = np.mean(v[mask])
    i_avg = np.mean(i_resampled[mask])

    # --- 5. Cálculo da impedância de surto
    Zs = np.abs(v_avg / i_avg)

    print(f"✅ Impedância de surto (média {t_in_ns}-{t_fi_ns} ns): {Zs:.2f} Ohms")
    print(f"   Média V = {v_avg:.3f} V, Média I = {i_avg:.3e} A")

    return Zs

def extract_webdigitized_data(
    json_file_name: str,
    inputs_folder: Path = INPUTS
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê um arquivo JSON exportado pelo WebPlotDigitizer, extrai
    o primeiro dataset, ordena por X e retorna dois arrays: X e Y.
    
    Parâmetros
    ----------
    json_file_name : str
        Nome do arquivo JSON (deve estar em `inputs_folder`).
    inputs_folder : Path
        Pasta onde está o JSON.
    
    Retorno
    -------
    x, y : tuple de np.ndarray
        Vetores com as coordenadas ordenadas.
    """
    json_path = inputs_folder / json_file_name
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON não encontrado: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    
    data_points = obj['datasetColl'][0]['data']
    # ordena pelo valor calibrado em X
    data_sorted = sorted(data_points, key=lambda pt: pt['value'][0])
    
    x = np.array([pt['value'][0] for pt in data_sorted])
    y = np.array([pt['value'][1] for pt in data_sorted])
    return x, y

def plot_excitation_file(
    json_file_name: str,
    exc_name: str = f"{CASE_NAME}.exc"
) -> None:
    """
    Sobrepõe o arquivo .exc gerado pela simulação com os pontos
    extraídos do JSON do WebPlotDigitizer (Figura 5 de Noda).
    """
    # extrai pontos de referência
    x_ref, y_ref = extract_webdigitized_data(json_file_name)
    
    # carrega .exc da simulação
    exc_path = EXCITATIONS / exc_name
    if not exc_path.is_file():
        raise FileNotFoundError(f"Arquivo de excitação não encontrado: {exc_path}")
    data = np.loadtxt(exc_path)
    exc_time    = data[:, 0] * 1e9  # converte para ns
    exc_voltage = data[:, 1]
    
    # plot
    plt.figure(figsize=(8, 4))
    plt.plot(exc_time,    exc_voltage, label=exc_name,   color='blue')
    plt.scatter(x_ref,     y_ref,        s=15, marker='o',
                label='Noda (2002)',   color='red')
    plt.xlabel('Tempo (ns)')
    plt.ylabel('Tensão (V)')
    plt.title('Fig. 6a. Waveform of voltage source')
    plt.xlim(-10, 40)
    plt.ylim(-10, 100)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_wire_data_current(
    probe_name: str,
    json_file_name: str
) -> None:
    """
    Sobrepõe a corrente medida na sonda de fio (simulação)
    com os pontos extraídos do JSON do WebPlotDigitizer (Fig. 6c).
    Ignora o último valor da série simulada.
    """
    # extrai pontos de referência
    x_ref, y_ref = extract_webdigitized_data(json_file_name)
    
    # carrega dados de corrente da simulação
    data_dict = extract_probe_wire_data(OUTPUTS)
    if not data_dict:
        raise RuntimeError(f"Nenhum arquivo de corrente encontrado em '{OUTPUTS}'")
    if probe_name not in data_dict:
        raise ValueError(f"Probe '{probe_name}' não encontrada. Escolha entre {list(data_dict.keys())}")
    
    time_full = data_dict[probe_name]['time']
    curr_full = data_dict[probe_name]['current']
    # descarta último ponto e converte para ns
    time = time_full[:-1] * 1e9
    curr = curr_full[:-1]
    
    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(time, curr,                  label=probe_name, color='blue')
    plt.scatter(x_ref, y_ref, s=15,       marker='o',
                label='Noda (2002)',      color='red')
    plt.xlabel('Tempo (ns)')
    plt.ylabel('Corrente (A)')
    plt.title('Fig. 6c. Current in wire probe')
    plt.xlim(-10, 40)
    plt.ylim(-0.1, 0.3)
    plt.xticks(np.arange(-10, 41, 5))
    plt.yticks(np.arange(-0.1, 0.31, 0.05))
    plt.grid(False)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    # Garante a existência dos diretórios de saída e logs
    ensure_folders_exist(OUTPUTS, LOGS, EXCITATIONS)

    # Copia examples\{CASE_NAME}\{CASE_NAME}\ugrfdtd para examplesData\cases
    copy_json_file_from_ugrfdtd()

    # Cria o arquivo de excitação com base no JSON
    create_excitation_file()

    # # Executa a simulação e obtém os arquivos de saída das sondas
    # run_probes = run_simulation()

    # # Plota a curva de excitação
    plot_excitation_file(json_file_name='noda2002_fig_6a.json')

    # # Plota os resultados das sondas de corrente
    # probe_wire_data = extract_probe_wire_data()
    # print(f"🔍 Probe_wire_data: {list(probe_wire_data.keys())}")
    # plot_probe_wire_data('current')

    # # Plota as curvas de corrente das sondas Wx, Wy e Wz
    plot_wire_data_current(
        probe_name='noda2002.fdtd_Wire_probe_A_Wz_12_12_2_s1.dat',
        json_file_name='noda2002_fig_6c.json')

    # # Plota a comparação entre a excitação e E_dl
    # plot_excitation_vs_Edl()
