r"""
cmd commands:
    conda activate semba-fdtd
    cd C:\Users\adilt\OneDrive\05_GIT\openSEMBA\fdtd
    python examples/SphericalCavity.py

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
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, windows, find_peaks
from scipy.constants import c
from scipy.special import spherical_jn
from scipy.optimize import root_scalar
from typing import Optional, Literal, List, Tuple, Dict

# Insere o diretório atual no início da lista sys.path, com prioridade 0.
sys.path.insert(0, os.getcwd())
from src_pyWrapper.pyWrapper import FDTD, Probe

# Pastas principais
CASE_NAME   = 'SphericalCavity'
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

    # Garante a existência dos diretórios de saída e logs
    ensure_folders_exist(OUTPUTS, LOGS, EXCITATIONS)

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

def plot_excitation_file(exc_name: str = f"{CASE_NAME}.exc") -> None:
    """
    Plota a curva de tensão do arquivo .exc.

    Parâmetros
    ----------
    exc_name : str, opcional
        Nome do arquivo de excitação. Por padrão, 'NodaVoltage.exc'.

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

def plot_point_probe_fields(
    point_groups: dict[str, dict[str, pd.DataFrame]],
    field_coordinate: Optional[str] = None
) -> None:
    """
    Plota os campos E total e incidente para cada direção de cada sonda.

    Parâmetros
    ----------
    point_groups : dict
        Estrutura { nome_da_sonda : { direção : DataFrame } }.
    field_coordinate : {'x','y','z'}, opcional
        Se fornecido, plota apenas essa coordenada (Ex, Ey ou Ez).
        Se None, plota as três coordenadas como antes.
    """
    all_axes = ('x', 'y', 'z')
    # Determina quais eixos plotar
    if field_coordinate is None:
        axes = list(all_axes)
    else:
        fc = field_coordinate.lower()
        if fc not in all_axes:
            raise ValueError(f"field_coordinate deve ser um de {all_axes}, não '{field_coordinate}'")
        axes = [fc]

    for probe_name, dirs in point_groups.items():
        # cria uma linha por eixo selecionado
        fig, axs = plt.subplots(len(axes), 1, sharex=True,
                                figsize=(8, 4 * len(axes)))
        # garante que axs seja sempre iterável
        if len(axes) == 1:
            axs = [axs]

        for idx, axis in enumerate(axes):
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

def spherical_bessel_zeros(
    n: int,
    num_zeros: int,
    which: Literal['j','j_prime']='j',
    tol: float=1e-12,
    dx: float=0.1
) -> np.ndarray:
    """
    Encontra os primeiros `num_zeros` zeros positivos de j_n(x) ou j_n'(x)
    pela estratégia de varredura + bissecção.

    Parâmetros
    ----------
    n : int
        Ordem do spherical Bessel j_n.
    num_zeros : int
        Quantos zeros positivos coletar (p=1,2,...).
    which : {'j','j_prime'}
        'j'       → zeros de j_n(x)
        'j_prime' → zeros de j_n'(x)
    tol : float
        Tolerância para a bissecção.
    dx : float
        Passo de varredura em x para detectar mudança de sinal.

    Retorna
    -------
    np.ndarray
        Array com os `num_zeros` menores zeros positivos em ordem crescente.
    """
    # define a função alvo
    if which == 'j':
        f = lambda x: spherical_jn(n, x)
    elif which == 'j_prime':
        f = lambda x: spherical_jn(n, x, derivative=True)
    else:
        raise ValueError("which deve ser 'j' ou 'j_prime'")

    roots = []
    x = 1e-6  # ponto de partida (evita x=0 quando j_n(0)=0)
    # Um teto razoável para buscar zeros:
    x_max = (num_zeros + n/2 + 1) * np.pi

    while len(roots) < num_zeros and x < x_max:
        x_next = x + dx
        if f(x) * f(x_next) < 0:
            # há mudança de sinal → bracket encontrado
            sol = root_scalar(f, bracket=[x, x_next], method='bisect', xtol=tol)
            if not sol.converged:
                raise RuntimeError(f"Não convergiu na bissecção para a raiz {len(roots)+1} de ordem {n}")
            root = sol.root
            roots.append(root)
            # pula à frente para não voltar a encontrar o mesmo zero
            x = root + dx
            continue
        x = x_next

    if len(roots) < num_zeros:
        raise RuntimeError(f"Só encontrei {len(roots)} zeros (queria {num_zeros}) até x={x_max}")

    return np.array(roots)

def spherical_bessel_derivative_zeros(
    n: int,
    num_zeros: int,
    tol: float = 1e-12,
    dx: float = 1e-2
) -> np.ndarray:
    """
    Encontra os primeiros `num_zeros` zeros positivos de j_n'(x).
    """
    def f(x: float) -> float:
        return spherical_jn(n-1, x) - (n+1)/x * spherical_jn(n, x)

    roots = []
    x = dx
    x_max = (num_zeros + n/2 + 1) * np.pi

    while len(roots) < num_zeros and x < x_max:
        x_next = x + dx
        if f(x) * f(x_next) < 0:
            sol = root_scalar(f, bracket=[x, x_next], method='bisect', xtol=tol)
            if not sol.converged:
                raise RuntimeError(f"Falha na bissecção para zero #{len(roots)+1} de j_{n}'")
            roots.append(sol.root)
            x = sol.root + dx
            continue
        x = x_next

    if len(roots) < num_zeros:
        raise RuntimeError(
            f"Encontrados apenas {len(roots)} zeros (queria {num_zeros}) até x={x_max:.1f}"
        )

    return np.array(roots)

def plot_frequency_spectrum_components(
    point_data: Dict[str, Dict[str, pd.DataFrame]],
    probe_name: str,
    window_type: str = 'hann',
    show_psd: bool = False
) -> None:
    """
    Plota um único gráfico com as três componentes vetoriais (Ex, Ey, Ez) sobrepostas.

    Parâmetros
    ----------
    point_data : dict[str, dict[str, DataFrame]]
        Dicionário { nome_sonda : { 'x': df_x, 'y': df_y, 'z': df_z } }.
    probe_name : str
        Chave em point_data para escolher a sonda.
    window_type : str
        Nome da janela em scipy.signal.windows (e.g. 'hann', 'hamming').
    show_psd : bool
        Se True, plota PSD (|FFT|^2) em escala semilog-y; senão plota |FFT|.
    """
    data = point_data.get(probe_name)
    if data is None:
        raise KeyError(f"Sonda '{probe_name}' não encontrada em point_data")

    # Encontra qualquer componente disponível para extrair o vetor de tempo
    for comp in ('x', 'y', 'z'):
        if comp in data:
            df_sample = data[comp]
            break
    else:
        raise ValueError(f"Nenhuma componente 'x','y' ou 'z' encontrada para sonda '{probe_name}'")

    t = df_sample['time'].values
    dt = t[1] - t[0]
    N = len(t)
    freqs = np.fft.rfftfreq(N, dt)

    plt.figure(figsize=(8, 5))
    for comp in ('x', 'y', 'z'):
        if comp not in data:
            continue

        df = data[comp]
        e = df['field'].values

        # Remove tendência e aplica janela
        e_dt = detrend(e)
        win = getattr(windows, window_type)(N)
        e_win = e_dt * win

        # FFT
        E_f = np.fft.rfft(e_win)
        mag = np.abs(E_f)
        psd = mag**2

        label = f"E{comp.upper()}"
        if show_psd:
            plt.semilogy(freqs, psd, label=f"PSD {label}")
        else:
            plt.plot(freqs, mag, label=f"|{label}(f)|")

    plt.xlabel("Frequência [Hz]")
    plt.ylabel("PSD [arb.u.]" if show_psd else "|E(f)| [arb.u.]")
    plt.title(f"Sonda '{probe_name}' – Espectro de Frequência (Ex, Ey, Ez)")
    plt.legend()
    plt.xlim(0, 4*1e9)  # Limite superior de frequência (4 GHz)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
def calculate_resonant_frequencies(
    point_data: dict[str, dict[str, pd.DataFrame]],
    probe_name: str,
    direction: Optional[str] = None,
    n_peaks: int = 5,
    window_type: str = 'hann',
    peak_threshold: float = 0.1
) -> np.ndarray:
    """
    Calcula as frequências ressonantes a partir dos dados de campo elétrico no domínio temporal.

    Parâmetros
    ----------
    point_data : dict[str, dict[str, DataFrame]]
        Dicionário { nome_sonda : { 'x': df_x, 'y': df_y, 'z': df_z } } contendo as séries temporais.
    probe_name : str
        Nome da sonda a ser analisada (chave de point_data).
    direction : {'x','y','z'} ou None
        Se 'x','y' ou 'z', usa aquela componente;
        se None (padrão), usa o módulo |E| = sqrt(Ex² + Ey² + Ez²).
    n_peaks : int
        Número de picos (modos) ressonantes a retornar. Default 5.
    window_type : str
        Tipo de janela para suavização antes da FFT ('hann', 'hamming', etc.).
    peak_threshold : float
        Fração do máximo do espectro usada como limiar para detectar picos.

    Retorna
    -------
    res_freqs : np.ndarray
        Array com as frequências ressonantes identificadas, em Hz.
    """
    # 1) Extrai tempo e série de campo
    #    Se direção especificada, pega apenas aquela componente
    if direction in ('x', 'y', 'z'):
        df = point_data[probe_name][direction]
        t = df['time'].values
        e = df['field'].values
    else:
        # usa o módulo do campo: sqrt(Ex^2 + Ey^2 + Ez^2)
        df_x = point_data[probe_name]['x']
        df_y = point_data[probe_name]['y']
        df_z = point_data[probe_name]['z']
        t = df_x['time'].values
        Ex = df_x['field'].values
        Ey = df_y['field'].values
        Ez = df_z['field'].values
        e = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # 2) Calcula intervalo de amostragem
    dt = t[1] - t[0]

    # 3) Remove tendência e aplica janela
    e_dt = detrend(e)
    N = len(e_dt)
    win = getattr(windows, window_type, windows.hann)(N)
    e_win = e_dt * win

    # 4) FFT e espectro de potência
    E_f   = np.fft.rfft(e_win)
    freqs = np.fft.rfftfreq(N, dt)
    psd   = np.abs(E_f)**2

    # 5) Detecta picos acima de limiar
    peaks, props = find_peaks(psd, height=psd.max() * peak_threshold)

    # 6) Ordena por intensidade e escolhe os n_peaks maiores
    heights = props['peak_heights']
    idx_sort = np.argsort(heights)[::-1]
    top_idx = peaks[idx_sort][:n_peaks]

    # 7) Retorna as frequências correspondentes
    return freqs[top_idx]

def compute_all_resonant_modes(
    point_data: Dict[str, Dict[str, pd.DataFrame]],
    n_peaks: int = 5,
    window_type: str = 'hann',
    peak_threshold: float = 0.1
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Varre todas as sondas em point_data e calcula os modos ressonantes para
    cada direção ('x','y','z') ou para o módulo do campo se houver todas as três.

    Retorna um dicionário:
        { nome_sonda: { direção: array_de_freqs_resonantes } }
    """
    modes: Dict[str, Dict[str, np.ndarray]] = {}

    for pname, dirs in point_data.items():
        modes[pname] = {}
        for dirc in dirs:
            # chama calculate_resonant_frequencies com a direção apropriada
            freqs = calculate_resonant_frequencies(
                point_data,
                probe_name=pname,
                direction=dirc,
                n_peaks=n_peaks,
                window_type=window_type,
                peak_threshold=peak_threshold
            )
            modes[pname][dirc] = freqs

        # se tivermos todas as três componentes x,y,z, também podemos calcular o módulo
        if all(ax in dirs for ax in ('x','y','z')):
            modes[pname]['magnitude'] = calculate_resonant_frequencies(
                point_data,
                probe_name=pname,
                direction=None,
                n_peaks=n_peaks,
                window_type=window_type,
                peak_threshold=peak_threshold
            )

    return modes

def spherical_cavity_resonant_freqs(
    radius: float,
    mode: Literal['TE','TM'] = 'TM',
    order_max: int = 3,
    n_roots: int = 3
) -> dict[tuple[int,int], float]:
    """
    Calcula as frequências ressonantes de uma cavidade esférica PEC de raio `radius`.

    Parâmetros
    ----------
    radius : float
        Raio da cavidade (m).
    mode : {'TE','TM'}
        Tipo de modo:
          - 'TE': zeros de j_l(x) → modos TE_{l,n}.
          - 'TM': zeros de j_l'(x) → modos TM_{l,n}.
        (Harrington, Time‐Harmonic Electromagnetic Fields, Cap.6)
    l_max : int
        Valor máximo de l a considerar (l = 1…l_max).
    n_roots : int
        Número de raízes (n = 1…n_roots) de cada função.

    Retorna
    -------
    freqs : dict[(l,n), f]
        Dicionário cujas chaves são pares (l,n) e valores as frequências ressonantes f_{l,n} em Hz.
    """
    freqs: dict[tuple[int,int], float] = {}

    for order in range(1, order_max + 1):
        # escolhe a função de zero apropriada
        if mode.upper() == 'TE':
            zeros = spherical_bessel_zeros(n=order, num_zeros=n_roots)
        elif mode.upper() == 'TM':
            zeros = spherical_bessel_derivative_zeros(n=order, num_zeros=n_roots)
        else:
            raise ValueError("mode deve ser 'TE' ou 'TM'")

        # converte cada raiz x_{l,n} em frequência f_{l,n} = (c/2π)·(x_{l,n}/radius)
        for n, x_ln in enumerate(zeros, start=1):
            k_ln = x_ln / radius
            f_ln = (c / (2 * np.pi)) * k_ln
            freqs[(order, n)] = f_ln

    return freqs

def lowest_n_resonant_modes(
    radius: float,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Retorna as `top_n` menores frequências ressonantes (TE ou TM) de uma cavidade
    esférica PEC, indicando também o modo correspondente.

    Parâmetros
    ----------
    radius : float
        Raio da cavidade em metros.
    l_max : int
        Valor máximo de l a considerar (1…l_max).
    n_roots : int
        Número de raízes (n) de cada l a calcular (para TE e para TM).
        Deve valer: 2 * l_max * n_roots >= top_n.
    top_n : int
        Quantas menores frequências (TE ou TM) retornar (padrão 5).

    Retorna
    -------
    List[Tuple[str, float]]
        Lista de tuplas `(modo, frequência)` onde `modo` é uma string
        como "TE_{2,1}" ou "TM_{1,3}" e `frequência` está em Hz,
        ordenadas das menores às maiores.
    """
    # 1) gera modos TE e TM
    te_modes = spherical_cavity_resonant_freqs(
        radius=radius, mode='TE', order_max=top_n+1, n_roots=top_n
    )
    tm_modes = spherical_cavity_resonant_freqs(
        radius=radius, mode='TM', order_max=top_n+1, n_roots=top_n
    )

    # 2) une e formata em lista [(label, f), ...]
    all_modes: List[Tuple[str, float]] = []
    for (l, n), freq in te_modes.items():
        all_modes.append((f"TE_{{{l},{n}}}", freq))
    for (l, n), freq in tm_modes.items():
        all_modes.append((f"TM_{{{l},{n}}}", freq))

    # 3) ordena por frequência e retorna os top_n
    all_modes.sort(key=lambda item: item[1])
    return all_modes[:top_n]

if __name__ == '__main__':
    # Copia os arquivos de examples\{CASE_NAME}\{CASE_NAME}\ugrfdtd para
    # o diretório examplesData\cases
    # copy_json_file_from_ugrfdtd()

    # Cria o arquivo de excitação com base no JSON
    # create_excitation_file()

    # # Executa a simulação e obtém os arquivos de saída das sondas
    # run_probes = run_simulation()
    # print(f"🔍 Probes encontrados: {list(run_probes.keys())}")

    # Plota a curva de excitação
    plot_excitation_file(exc_name="predefinedExcitation.1.exc")

    # Carrega os arquivos de saída das sondas do diretório de saída
    run_probes = load_probes_from_outputs()
    print(f"🔍 Probes encontrados: {list(run_probes.keys())}")
    
    # Obtém os dados das sondas pontuais de E-field
    point_data = extract_point_probe_fields(run_probes)

    # # Calcula as frequências ressonantes para cada sonda
    # lowest_n = lowest_n_resonant_modes(
    #     radius=0.500,
    #     top_n=10
    # )
    # print("\nMenores frequências de ressonância:")
    # for mode_label, freq in lowest_n:
    #     print(f"{mode_label}: {freq/1e9:.3f} GHz")

    # # Calcula as frequências ressonantes numericamente
    # all_modes = compute_all_resonant_modes(
    #     point_data,
    #     n_peaks=3,
    #     window_type='hann',
    #     peak_threshold=0.1
    # )
    # for probe, dirs in all_modes.items():
    #     for dirc, freqs in dirs.items():
    #         print(f"{probe} ({dirc}): {freqs/1e9} GHz")        
    
    # Plotar os resultados das sondas
    plot_point_probe_fields(point_data, field_coordinate='z')

    # Plotar o espectro de Ex, Ey e Ez da sonda 'probe1', mostrando PSD:
    plot_frequency_spectrum_components(point_data, probe_name='Point probe')
    
    print("✅ Simulação concluída com sucesso.")