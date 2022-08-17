# Author: Fábio Campos
# Código principal para rodar o aplicativo

# verifica se as bibliotecas estão instaladas
import instalation

try:
    import kivy
except:
    # se não, instalá elas
    # instalation.install_kivy() #Pecesa descobrir como responder perguntas durante a instalação como sudo (senha) e y (yes)
    print("Instale a biblioteca Kivy !")


from kivy.core.camera import Camera
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen


class TelaPrincipal(ScreenManager):
    pass


class TelaDeConfiguracao(Screen):
    pass


class Qual_o_valor(App):
    def build(self):
        return TelaPrincipal()


if __name__ == '__main__':
    Qual_o_valor().run()
