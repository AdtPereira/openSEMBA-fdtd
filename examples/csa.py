r"""
cmd commands:
conda activate semba-fdtd
cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
python examples/csa.py

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
"""
# pylint: disable=unused-import, wrong-import-position

import os
import sys
import shutil
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.constants import c

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import FDTD, Probe

# Pastas principais
CASE_NAME   = 'csa'
SEMBA_EXE   = Path.cwd() / 'build' / 'bin' / 'semba-fdtd.exe'
EXAMPLES    = (Path.cwd() / 'examples').resolve()
LOGS        = (Path.cwd() / 'examplesData' / 'logs').resolve()
CASES       = (Path.cwd() / 'examplesData' / 'cases').resolve()
EXCITATIONS = (Path.cwd() / 'examplesData' / 'excitations').resolve()
OUTPUTS     = (Path.cwd() / 'examplesData' / 'outputs' / CASE_NAME).resolve()
INPUTS      = (Path.cwd() / 'examplesData' / 'inputs' / CASE_NAME).resolve()
JSON_FILE   = CASES / f'{CASE_NAME}.fdtd.json'

def ensure_folders_exist(
        *folders: list[Path]) -> None:
    """
    Cria os diretórios especificados se não existirem.
    Se o diretório já existir, não faz nada.
    """
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

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

def create_excitation_file(
    json_file_name: str,
    auto_plot: bool = True,
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

    # --- 3. Extrai pontos de referência
    baba_time, baba_current = extract_webdigitized_data(json_file_name)

    # --- 4. Interpolação linear
    dt = case_data["general"]["timeStep"]
    steps = case_data["general"]["numberOfSteps"]
    
    # --- 5. Cria o pulso gaussiano
    beta = 2.5      # adimensional
    tau0 = 100e-9   # tempo central (s)
    I0 = 1.0e3      # pico da corrente, em A
    z = 0.0         # distância do ponto de medição (m)
    delay = z / c   # atraso (s)
    time = np.arange(0, steps * dt, dt)
    gaussian_pulse = I0 * np.exp(- (beta/tau0)**2 * (time - tau0 - delay)**2)
    
    # --- 5. Salva o arquivo
    output = CASES / exc_name
    data = np.column_stack((time, gaussian_pulse))
    np.savetxt(output, data, fmt="%.8e", delimiter=' ')
    print(f"✅ Arquivo de excitação '{exc_name}' criado com sucesso em: {output}")

    # --- 6. Plota o resultado
    if auto_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(time*1e9, gaussian_pulse*1e-3, label=exc_name)
        plt.scatter(baba_time*1e9, baba_current*1e-3, s=12, marker='o', label='Baba (2003)', color='red')
        plt.xlabel('Time (ns)')
        plt.ylabel('Current (kA)')
        plt.title('Fig. 3a. The Gaussian current waveform at z'' = 0')
        plt.xlim(0, 1000)
        plt.ylim(0, 1.2)
        plt.xticks(np.arange(0, 1001, 200))
        plt.yticks(np.arange(0, 1.21, 0.2))
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()

def copy_json_file_from_ugrfdtd(
) -> None:
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

def run_simulation(
) -> dict[str, str]:
    """Run the simulation and return o dicionário de probes com caminhos absolutos."""

    if not SEMBA_EXE.is_file():
        raise FileNotFoundError(SEMBA_EXE)
    if not JSON_FILE.is_file():
        raise FileNotFoundError(JSON_FILE)

    solver = FDTD(input_filename=str(JSON_FILE.resolve()),
              path_to_exe=str(SEMBA_EXE),
              flags=['-stableradholland'])
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

        # 5) Move arquivos .dat para OUTPUTS
        if fname.endswith('.dat'):
            dst = (OUTPUTS / fname).resolve()
            if src.exists():
                os.replace(str(src), str(dst))
                print(f"📊 Arquivo de saída '{fname}' movido para '{OUTPUTS}'.")
                probes[fname] = str(dst)
            else:
                print(f"⚠ Aviso: '{fname}' não encontrado. Pulando.")

        # 6) Move arquivos .exc para EXCITATIONS
        if fname.endswith('.exc'):
            dst = (EXCITATIONS / fname).resolve()
            if src.exists():
                os.replace(str(src), str(dst))
                print(f"📡 Arquivo de excitação '{fname}' movido para '{EXCITATIONS}'.")
            else:
                print(f"⚠ Aviso: '{fname}' não encontrado. Pulando.")

    return probes

def load_probes_from_outputs(
) -> dict[str, str]:
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

def extract_point_probe_data(
        probes: dict[str, str]
) -> dict[str, dict[str, pd.DataFrame]]:
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

def extract_wire_probe_data(
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
    wire_data: Dict[str, Dict[str, np.ndarray]] = {}

    for fname in os.listdir(OUTPUTS):
        if fname.endswith('.dat') and 'fdtd_Wire probe' in fname:
            path = os.path.join(OUTPUTS, fname)
            # carrega tudo, pulando o header de uma linha
            arr = np.loadtxt(path, skiprows=1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            wire_data[fname] = {
                'time'          : arr[:, 0],
                'current'       : arr[:, 1],
                'E_dl'          : arr[:, 2],
                'Vplus'         : arr[:, 3],
                'Vminus'        : arr[:, 4],
                'Vplus-Vminus'  : arr[:, 5],
            }

    return wire_data

def extract_bulk_probe_data(
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
    bulk_data: Dict[str, Dict[str, np.ndarray]] = {}

    for fname in os.listdir(OUTPUTS):
        if fname.endswith('.dat') and 'fdtd_Bulk probe' in fname:
            path = os.path.join(OUTPUTS, fname)
            # carrega tudo, pulando o header de uma linha
            arr = np.loadtxt(path, skiprows=1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            bulk_data[fname] = {
                'time'          : arr[:, 0],
                'current'       : arr[:, 1],
            }

    return bulk_data

def plot_vector_field_data(
    run_probes: dict[str, str]
) -> None:
    """
    Plota, em um único gráfico, as componentes Ex, Ey e Ez do campo elétrico
    (total e incidente, se houver) para todas as sondas disponíveis.

    Se ambas as séries (total e incidente) de uma componente estiverem
    completamente nulas, a curva não é plotada.

    Parâmetros
    ----------
    run_probes : dict[str, str]
        Mapeamento { nome_da_sonda : caminho_para_arquivo } usado por
        extract_point_probe_data().
    """
    line_styles = {'x': ':', 'y': '--', 'z': '-'}
    point_groups = extract_point_probe_data(run_probes)
    plt.figure(figsize=(10, 6))
    for probe_name, dirs in point_groups.items():
        for axis, style in line_styles.items():
            if axis not in dirs:
                continue

            df = dirs[axis]
            f = abs(df['field'])*1e-3
            inc = df['incident'] if 'incident' in df.columns else None

            # verifica se tanto 'field' quanto 'incident' são todos zeros
            zero_field    = not f.any()
            zero_incident = (inc is None) or (not inc.any())
            if zero_field and zero_incident:
                continue

            t_ns = df['time'] * 1e9
            # plota campo total, se não todo zero
            if not zero_field:
                plt.plot(
                    t_ns,
                    f,
                    linestyle=style,
                    linewidth=1.5,
                    label=fr"{probe_name} $E_{axis.upper()}$"
                )

            # plota campo incidente, se presente e não todo zero
            if inc is not None and not zero_incident:
                plt.plot(
                    t_ns,
                    inc,
                    linestyle=style,
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"{probe_name} E{axis.upper()} incident"
                )

    plt.xlabel("Time [ns]")
    plt.ylabel("Electric Field [kV/m]")
    plt.title("Fig. 6b. Waveforms of electric field at different heights")
    plt.xlim(0, 1000)
    plt.ylim(0, 2)
    plt.xticks(np.arange(0, 1001, 200))
    plt.yticks(np.arange(0, 2.1, 0.5))
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()

def plot_current_data(
    json_file_name: str = None
) -> None:
    """
    Sobrepõe a corrente medida na sonda de fio (simulação)
    com os pontos extraídos do JSON do WebPlotDigitizer (Fig. 6c).
    Ignora o último valor da série simulada.
    """
    # carrega dados de wire probes
    bulk_dict = extract_bulk_probe_data()
    if not bulk_dict:
        raise RuntimeError(f"Nenhum arquivo de corrente encontrado em '{OUTPUTS}'")

    # plot
    plt.figure(figsize=(8, 5))
    # for wire_name, d in wire_dict.items():
    #     plt.plot(d['time'][:-1]*1e9, d['current'][:-1], label=wire_name.replace('.dat', ''))
    
    for bulk_name, d in bulk_dict.items():
        plt.plot(d['time'][:-1]*1e9, d['current'][:-1]*1e-3, label=bulk_name.replace('.dat', ''))

    if json_file_name is not None:
        baba_time, baba_current = extract_webdigitized_data(json_file_name)
    
    plt.xlabel('Time (ns)')
    plt.ylabel('Current (kA)')
    plt.title('Fig. 6a. Current waveforms on the vertical conductor observed at different heights')
    plt.xlim(0, 1000)
    plt.ylim(0, 1.2)
    plt.xticks(np.arange(0, 1001, 200))
    plt.yticks(np.arange(0, 1.21, 0.2))
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_voltage_data(
    json_file_name: str,
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
    # 1) extrai pontos de referência
    noda_time_ns, noda_voltage = extract_webdigitized_data(json_file_name)

    # 2) Carrega E_dl de cada sonda
    wire_data: Dict[str, Dict[str, np.ndarray]] = extract_wire_probe_data()
    if not wire_data:
        raise RuntimeError(f"Nenhum arquivo de fio encontrado em '{OUTPUTS}'")

    plt.figure(figsize=(8, 5))
    plt.scatter(noda_time_ns, noda_voltage, s=12, marker='o', label='Noda (2002)', color='red')
    for fname, cols in wire_data.items():
        plt.plot(cols['time'][:-1]*1e9, cols['E_dl'][:-1], label=fname.replace('.dat', ''))

    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.title('Fig. 6b. Voltage waveform in wire probe')
    plt.xlim(0, 40)
    plt.ylim(-20, 80)
    plt.xticks(np.arange(0, 41, 10))
    plt.yticks(np.arange(-20, 81, 20))
    plt.grid(False)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def compute_surge_impedance(
    t_start_ns: float = 15.0,
    t_end_ns: float = 20.0
) -> dict[str, float]:
    """
    Calcula a razão média entre E_dl e corrente no intervalo de tempo especificado.

    Parâmetros
    ----------
    t_start_ns : float
        Início do intervalo de tempo (em ns).
    t_end_ns : float
        Fim do intervalo de tempo (em ns).

    Retorno
    -------
    Dict[str, float]
        Dicionário com os nomes das sondas e suas respectivas razões médias V/I (Ohms).
    """
    data_dict = extract_wire_probe_data()
    if not data_dict:
        raise RuntimeError("Nenhum arquivo de corrente foi encontrado.")

    zs = {}
    for fname, data in data_dict.items():
        t_ns = data['time'] * 1e9
        mask = (t_ns >= t_start_ns) & (t_ns <= t_end_ns)

        # Verifica se há pontos suficientes no intervalo
        if not np.any(mask):
            continue

        abs_mean_voltage = abs(np.mean(data['E_dl'][mask]))
        abs_mean_current = abs(np.mean(data['current'][mask]))

        if np.isclose(abs_mean_current, 0):
            zs[fname] = np.nan
        else:
            zs[fname] = abs_mean_voltage / abs_mean_current

        # Saída de dados
        print(f"Arquivo: {fname}")
        print(f"  V_avg: {abs_mean_voltage:.2f} V")
        print(f"  I_avg: {abs_mean_current:.2f} A")
        print(f"    Z_s: {zs[fname]:.0f} Ohms")

    return zs

if __name__ == '__main__':
    ## Copia examples\{CASE_NAME}\{CASE_NAME}\ugrfdtd para examplesData\cases
    ensure_folders_exist(OUTPUTS, LOGS, EXCITATIONS)
    copy_json_file_from_ugrfdtd()

    ## Cria o arquivo de excitação com base no JSON
    create_excitation_file(
        json_file_name='baba2003_fig_3a.json', auto_plot=False)
    run_probes = run_simulation()

    ## Plota os resultados das sondas de tensão e corrente
    plot_current_data()
    plot_vector_field_data(run_probes)