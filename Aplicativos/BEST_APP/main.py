# Author: Fábio Campos
# Código principal para rodar o aplicativo

# verifica se as bibliotecas estão instaladas
# import instalation

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.utils import platform
from PIL import Image
import processamentoDeImagem


def classify_image(*largs):
    """??

    Args:
        None

    Returns:
        None
    """
    # print(largs)
    camera = self.ids['camera']
    pixels = camera.texture.pixels
    im = Image.frombytes('RGBA', (640, 480), pixels)
    # y = classifiy(im)
    y = 4
    #criar agora metodo para mostrar resultado

def resquests_for_android():
    """??

    Args:
        None

    Returns:
        None
    """
    if platform == "android":
        from android.permissions import request_permissions, Permission
        request_permissions([Permission.CAMERA])
        from android.permissions import check_permission
        while (not check_permission(Permission.CAMERA)):
            print("Esperando autorização da camera")


class TelaDaCamera(Screen):
    pass


class TelaDeConfiguracao(Screen):
    def inverteCamera(self):
        # self.manager.get_screen("tela_da_camera").ids.camera.index=int(not self.manager.get_screen("tela_da_camera").ids.camera.index)
        # print(self.manager.get_screen("tela_da_camera").ids.camera.index)
        pass


class GerenciadorDeTelas(ScreenManager):
    pass


class Qual_o_valor(App):
    def build(self):
        Clock.schedule_interval(classify_image, INTERVAL)
        return GerenciadorDeTelas()


if __name__ == '__main__':
    INTERVAL = 1
    resquests_for_android()
    classify_image()
    Qual_o_valor().run()
