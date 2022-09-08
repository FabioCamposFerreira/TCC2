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
