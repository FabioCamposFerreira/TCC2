# Author: Fábio Campos
# Código principal para rodar o aplicativo

# verifica se as bibliotecas estão instaladas
try:
    import instalation
    import kivy
# se não, instalá elas
except:
    instalation.libraries()

from kivy.core.camera import Camera
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget


class CameraApp(App):
    pass


if __name__ == '__main__':
    CameraApp().run()
