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


def install_buildozer():
    # Comandos parar usuários Linux
    # Instalando e confugurando buildozer
    # Baixando e instalando
    os.system("git clone https: // github.com/kivy/buildozer.git")
    os.system("cd buildozer")
    os.system("sudo python setup.py install")
    # Dentro da pasta do projeto, inicialize o buildozer
    os.system("buildozer init")
    # Edite o arquivo buildozer.spec
    # Instalando dependências do python-for-android ??não tenho certezas??
    os.system("sudo dpkg --add-architecture i386")
    os.system("sudo apt-get update")
    os.system("sudo apt-get install -y build-essential ccache git zlib1g-dev python3 python3-dev libncurses5:i386 libstdc++6:i386 zlib1g:i386 openjdk-8-jdk unzip ant ccache autoconf libtool libssl-dev")
    # Instale o cytron
    os.system("pip install Cython")
    # Corrigindo erro do Cython3
    os.system(
        "https://stackoverflow.com/questions/66973759/cython-not-found-please-install-it-error-in-buildozer")
    # Tratando erro No module named '_ctypes'
    os.system("sudo apt-get install libffi-dev")
    os.system("buildozer android debug deploy run")

    # https://buildozer.readthedocs.io/en/latest/quickstart.html
    # Comands to run buildozer
    # buildozer -v android deploy run logcat | grep python

def install_matplotlib():
    os.system("python3 -m pip install -U pip")
    os.system("python3 -m pip install -U matplotlib")