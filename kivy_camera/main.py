__version__ = '1.0'

import kivy

# importing file from https://github.com/kivy/plyer/blob/master/plyer/platforms/android/camera.py
# I downloaded it and saved it in the same directory:
from camera import AndroidCamera

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty

import base64

class MyCamera(AndroidCamera):
    pass

class BoxLayoutW(BoxLayout):
    my_camera = ObjectProperty(None)
    # /sdcard means internal mobile storage for that case:
    image_path = StringProperty('/sdcard/my_test_photo.png')

    def __init__(self, **kwargs):

        super(BoxLayoutW, self).__init__()

        self.my_camera = MyCamera()

    def take_shot(self):
        self.my_camera._take_picture(self.on_success_shot, self.image_path)

    def on_success_shot(self, loaded_image_path):
        # converting saved image to a base64 string:
        image_str = self.image_convert_base64
        return True

    #converting image to a base64, if you want to send it, for example, via POST:
    def image_convert_base64(self):
        with open(self.image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        if not encoded_string:
            encoded_string = ''
        return encoded_string

if __name__ == '__main__':

    class CameraApp(App):
        def build(self):
            main_window = BoxLayoutW()
            return main_window

    CameraApp().run()