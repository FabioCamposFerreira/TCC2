# Author: Fábio Campos
# Código principal para rodar o aplicativo

# verifica se as bibliotecas estão instaladas
# import instalation

from kivy.core.camera import Camera
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
import processamentoDeImagem
# from matplotlib import pyplot as plt
from kivy.utils import platform
if platform == "android":
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.CAMERA])
    from android.permissions import  check_permission         
    while (not check_permission(Permission.CAMERA)):
        print("Esperando autorização da camera")


class TelaDaCamera(Screen):
    def tira_foto(self):
        # camera = self.ids['camera']
        # camera.export_to_png("IMG_.png")
        # textura = camera.texture
        # pixels = texture.pixels
        # print("Captured")
        # im = processamentoDeImagem.open_image("IMG_.png")
        # im.show()
        # histograma = processamentoDeImagem.histogram("IMG_.png")
        # plt.plot(histograma)
        # plt.show()
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
        return GerenciadorDeTelas()

if __name__ == '__main__':
    Qual_o_valor().run()