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

from src_pyWrapper.pyWrapper import CaseMaker, Probe

EXCITATIONS_FOLDER = os.path.join('examplesData', 'excitations')
OUTPUTS_FOLDER = os.path.join('examplesData', 'outputs')
LOGS_FOLDER = os.path.join('examplesData', 'logs')
CASES_FOLDER = os.path.join('examplesData', 'cases')

def run_simulation():
    executable = os.path.join("build", "bin", "semba-fdtd.exe")
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"Executable {executable} not found.")

    input_file = os.path.join(CASES_FOLDER, 'holland1981.fdtd.json')

    # --- Copiar arquivo de excitação para a raiz (fdtd/)
    excitation_file = os.path.join(EXCITATIONS_FOLDER, 'holland.exc')
    if not os.path.isfile('holland.exc'):
        if os.path.isfile(excitation_file):
            shutil.copy2(excitation_file, '.')
        else:
            raise FileNotFoundError(f"Excitation file {excitation_file} not found.")

    print(f"Executando: {executable} -i {input_file}")
    subprocess.run([executable, "-i", input_file], check=True)
    print("Simulação finalizada.")

    # --- Criar pastas se necessário
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)

    # --- Mover arquivos
    for file in os.listdir(CASES_FOLDER):
        if file.startswith('holland1981'):
            src = os.path.join(CASES_FOLDER, file)
            if file.endswith('.dat') or file.endswith('.h5'):
                dst = os.path.join(OUTPUTS_FOLDER, file)
                shutil.move(src, dst)
            elif file.endswith('.txt'):
                dst = os.path.join(LOGS_FOLDER, file)
                shutil.move(src, dst)

    # --- Limpar o arquivo de excitação temporário
    if os.path.isfile('holland.exc'):
        os.remove('holland.exc')

def check_output_files(case_name, output_folder, expected_files):
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

if __name__ == "__main__":
    run_simulation()

    # --- Depois da simulação, confere os resultados ---
    expected_files = [
        'holland1981.fdtd_mid_point_Wz_11_11_12_s2.dat',
        'holland1981.fdtd_Energy.dat'
    ]
    check_output_files('holland1981', OUTPUTS_FOLDER, expected_files)

    print("\nAnálise de resultados concluída.")

