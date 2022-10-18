# Author: Fábio Campos
# Main code to run application

from plyer import tts
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.utils import platform


import image_processing
import platform
import classification as clsf


def requests_for_android():
    """Execute if running in android to request permissions
    """
    if platform == "android":
        from android.permissions import request_permissions, Permission
        request_permissions([Permission.CAMERA])
        request_permissions([Permission.WRITE_EXTERNAL_STORAGE])
        from android.permissions import check_permission
        while (not check_permission(Permission.CAMERA)):
            print("Esperando autorização da camera")
        while (not check_permission(Permission.WRITE_EXTERNAL_STORAGE)):
            print("Esperando autorização da escrever")


class ScreenCamera(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.event = Clock.schedule_interval(self.classify_image, 2)

    def classify_image(self, *largs):
        """Get and classify image
        """
        camera = self.ids['camera']
        texture = camera.export_as_image().texture
        im = image_processing.process_texture(texture)
        y = clsf.classify(im)
        valor = self.ids['valor']
        if y != None:
            valor.text = str(int(y))
            valor.color = (1, 1, 0, 1)
            print(str(platform))
            if platform == "android":
                tts.speak("".join((str(int(y))," reais")))
        else:
            valor.color = (1, 1, 0, 0)


class TelaDeConfiguracao(Screen):
    def camera_invert(self):
        camera = self.manager.get_screen("screen_camera").ids["camera"]
        camera.play = False
        camera.index = int(not (camera.index))
        camera.play = True


class GerenciadorDeTelas(ScreenManager):
    pass


class Qual_o_valor(App):
    def build(self):
        return GerenciadorDeTelas()


if __name__ == '__main__':
    if platform == "linux":
        import instalation
    elif platform == "Windows":
        # TODO
        pass
    requests_for_android()
    Qual_o_valor().run()
