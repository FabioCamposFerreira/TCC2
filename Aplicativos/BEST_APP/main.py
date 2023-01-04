# Author: Fábio Campos
# Main code to run application
import csv
import time
import cv2 as cv

from plyer import tts
from kivy.app import App
from kivy.clock import Clock
from kivy.utils import platform
from kivy.uix.screenmanager import ScreenManager, Screen

import image_processing
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
        self.labeling_status = False
        self.classes = [2, 5, 10, 20, 50, 100]
        self.correct_class = 0
        self.event = Clock.schedule_interval(self.classify_image, 2)

    def change_class(self):
        select_class = self.ids["select_class"]
        self.correct_class += 1
        if self.correct_class == 6:
            self.correct_class = 0
        select_class.text = "".join(("R$ ", str(self.classes[self.correct_class]), ",00"))

    def start_stop_labeling(self):
        start_stop = self.ids["start_stop"]
        self.labeling_status = not self.labeling_status
        if self.labeling_status:
            try:
                self.event()
            except:
                self.event = Clock.schedule_interval(self.classify_image, 2)
            start_stop.background_color = .37, .17, .59, 1
            start_stop.text = "Stop!"
        else:
            self.event.cancel()
            start_stop.background_color = .37, .17, .59, .5
            start_stop.text = "Start!"
            valor = self.ids['valor']
            valor.color = (1, 1, 0, 0)

    def classify_image(self, *largs):
        """Get and classify image
        """
        camera = self.ids['camera']
        texture = camera.export_as_image().texture
        im = image_processing.process_texture(texture)
        im_name = "".join((str(self.classes[self.correct_class]), ".", str(int(time.time())), ".png"))
        if platform == "android":
            # Save imgs to increse database
            cv.imwrite("".join(("/storage/emulated/0/Download/", im_name)), im)
        y = clsf.classify(im)
        valor = self.ids['valor']
        if y != None:
            valor.text = str(int(y))
            valor.color = (1, 1, 0, 1)
            if platform == "android":
                tts.speak("".join((str(int(y)), " reais")))
                # Save sequence of the results to mount table acuracy
                with open("".join(("/storage/emulated/0/Download/results", ".csv")), "a") as f:
                    writer = csv.writer(f)
                    # image name, predict, result
                    writer.writerow([im_name, int(y), self.classes[self.correct_class] == int(y)])
        else:
            valor.color = (1, 1, 0, 0)


class TelaDeConfiguracao(Screen):
    def camera_invert(self):
        camera = self.manager.get_screen("screen_camera").ids["camera"]
        camera.play = False
        camera.index = int(not (camera.index))
        camera.play = True
        pass


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
