# Author: Fábio Campos
# Executa linhas de comando para instalar bibliotecas de forma automática

import os


def install_pip():
    # Comandos parar usuários Linux
    # Instalando pip
    os.system("sudo apt update")
    os.system("sudo apt install python3-pip")
    os.system("pip install --upgrade pip")


def install_kivy():
    # Comandos parar usuários Linux
    # Instalando Kivy para a interface
    install_pip()
    os.system("echo \"Istalando Kivy\"")
    os.system("python3 -m pip install --upgrade pip setuptools virtualenv")
    os.system("python3 -m virtualenv kivy_venv")
    os.system("source kivy_venv/local/bin/activate")
    os.system("python3 -m pip install \"kivy[full]\" kivy_examples")


def install_opencv():
    # Comandos parar usuários Linux
    # Instalando OpenCV para a Camera
    install_pip()
    os.system("pip install opencv-python")


def install_numpy():
    # Comandos parar usuários Linux
    # Instalando Kivy para a interface
    os.system()
