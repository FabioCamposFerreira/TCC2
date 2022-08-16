from kivy.app import App
from kivy.uix.screenmanager import ScreenManager,Screen

class Interface(ScreenManager):
    pass
class Help(Screen):
    pass


class Qual_valor(App):
    def build(self):
        return Interface()


Qual_valor().run()
