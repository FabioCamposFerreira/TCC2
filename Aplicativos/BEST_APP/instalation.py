# Author: Fábio Campos
# Run commands to install python and os libraries to developers
import platform
import os
try:
    import kivy
except:
    install_kivy()
    # print("\033[92mInstale a biblioteca Kivy !\033[00m")
try:
    import numpy
    import cv2
    import PIL
except:
    install_pip()
    # print("\033[92m!\033[00m")


def install_pip():
    """Command to Linux
    Instalando pip"""
    os.system("sudo apt update")
    os.system("sudo apt install python3-pip")
    os.system("pip install --upgrade pip")


def install_libs():
    install_pip()
    os.system("pip install -r requirements.txt")


def install_kivy():
    """Command to Linux
    Installing Kivy"""
    install_pip()
    print("Installing Kivy")
    os.system("python3 -m pip install --upgrade pip setuptools virtualenv")
    os.system("python3 -m virtualenv kivy_venv")
    os.system("source kivy_venv/local/bin/activate")
    os.system("python3 -m pip install \"kivy[full]\" kivy_examples")


def install_buildozer():
    """Command to Linux
    Installing Buildozer"""
    # Commands
    # buildozer -v android debug deploy run logcat
    # clear && buildozer -v android debug deploy run logcat | grep python
    if platform.version().find("Ubunto"):
        os.system("pip3 install --user --upgrade buildozer")
        os.system("sudo apt update")
        os.system("sudo apt install -y git zip unzip openjdk-13-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev")
        os.system("pip3 install --user --upgrade Cython==0.29.19 virtualenv  # the --user should be removed if you do this in a venv")
        # add the following line at the end of your ~/.bashrc file
        # export PATH=$PATH:~/.local/bin/
    else:
        print("\033[92m[ERROR] Install buildoze fail!\033[00m")
        print("Use um sistema Ubunto ou WSL!")

## Google Store
# keytool -genkey -v -keystore key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias key
# password: Federalizacao1313
# What is your first and last name?
#   [Unknown]:  Fabio Ferreira    
# What is the name of your organizational unit?
#   [Unknown]:  Unversidade Federal de Urbelandia
# What is the name of your organization?
#   [Unknown]:  FEELT
# What is the name of your City or Locality?
#   [Unknown]:  Patos de Minas
# What is the name of your State or Province?
#   [Unknown]:  Minas Gerais
# What is the two-letter country code for this unit?
#   [Unknown]:  BR
# Is CN=Fabio Ferreira, OU=Unversidade Federal de Urbelandia, O=FEELT, L=Patos de Minas, ST=Minas Gerais, C=BR correct?
#   [no]:  yes

export P4A_RELEASE_KEYSTORE="key.jsk"
export P4A_RELEASE_KEYSTORE_PASSWD="Federalizacao1313"
export P4A_RELEASE_KEYALIAS_PASSWD="Federalizacao1313"
export P4A_RELEASE_KEYALIAS="key"


