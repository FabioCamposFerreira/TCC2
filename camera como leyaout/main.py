from kivy.core.camera import Camera
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager,Screen
#import classificacao
import time

class Interface(ScreenManager):
    def capture(self, *args, **kwargs):
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        return ("Captured")
        #  classificacao.classif(self)
        #  result = open('result.txt', 'r').readlines()
        #  return result
class Help(Screen):
    pass


class Qual_valor(App):
    def build(self, *args, **kwargs):
        return Interface()


Qual_valor().run()
