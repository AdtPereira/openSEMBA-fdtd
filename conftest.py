"""
Esse conftest.py faz duas coisas:
    Garante que build/bin está no PATH → seu semba-fdtd.exe sempre será encontrado.
    Garante que o diretório atual (fdtd/) está no sys.path → seus imports (src_pyWrapper) sempre funcionarão.
"""

import os
import sys

def pytest_configure(config):
    # Adiciona o diretório build/bin ao PATH se não estiver
    build_bin = os.path.join(os.getcwd(), 'build', 'bin')
    if build_bin not in os.environ["PATH"]:
        os.environ["PATH"] = build_bin + os.pathsep + os.environ["PATH"]

    # Garante que o diretório atual esteja no sys.path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
