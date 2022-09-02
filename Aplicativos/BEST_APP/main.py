# Author: Fábio Campos
# Código principal para rodar o aplicativo

# verifica se as bibliotecas estão instaladas
# import instalation

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.utils import platform
from PIL import Image
import classification as clsf
import image_processing

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


class ScreenCamera(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)
        Clock.schedule_interval(self.classify_image, 2)

    def classify_image(self, *largs):
        """Get and classify image
        """
        camera = self.ids['camera']
        texture = camera.export_as_image().texture
        im = image_processing.process_texture(texture)
        y = clsf.classifiy(im)
        valor = self.ids['valor']
        if y != None:
            valor.text = str(int(y))
            valor.color = (1, 1, 0, 1)
        else:
            valor.color = (1, 1, 0, 0)
        
        


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
    resquests_for_android()
    Qual_o_valor().run()
