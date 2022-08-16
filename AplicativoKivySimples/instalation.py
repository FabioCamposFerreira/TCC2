# Author: Fábio Campos
# Executa linhas de comando para instalar bibliotecas de forma automática

import os


def libraries():
    # Comandos parar usuários Linux
    # Instalando Kivy
    os.system("Istalando bibliotecas;")
    os.system("python3 -m pip install --upgrade pip setuptools virtualenv")
    os.system("python3 -m virtualenv kivy_venv")
    os.system("source kivy_venv/bin/activate")
    os.system("python -m pip install \"kivy[full]\" kivy_examples")
