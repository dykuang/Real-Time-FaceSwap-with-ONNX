from kivy.config import Config
# Config.set('kivy', 'exit_on_escape', '0')
# Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '640')
Config.set('graphics', 'height', '480')

import os
import cv2
from detection import Face_Detector, Landmark_Detector
from faceswap_cam import face_swap

from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock

from kivy.uix.label import Label
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen, ScreenManager

# from kivy.properties import ObjectProperty

import numpy as np

class MyScreenManager(ScreenManager):
    pass
class PreScreen(Screen):
    pass

class FrontScreen(Screen):
    def __init__(self, **kwargs):
        super(FrontScreen, self).__init__(**kwargs)
        self.refresh_dt = 0.05

    def on_enter(self, *args):  # only works for multiple screens?
        self.face_detector = Face_Detector()
        self.lmk_detector = Landmark_Detector()
        self.portraits = os.listdir('./portraits/')
        # print(self.portraits)
        
        '''
        dropdown menu
        '''
        self.dropdown = DropDown()
        for face in self.portraits:
            btn = Button(text=face, size_hint_y=None, height=32, 
                        #  color = (0,0,1,1),
                        #  background_normal='',background_color=(0.11, 0.17, 0.44, 0.2)
                         )

            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))

            self.dropdown.add_widget(btn)
        
        self.ids.face_selection.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self.ids.face_selection, 'text', x))

    def initialize(self, target_face):
        # self.face_detector = Face_Detector()
        # self.lmk_detector = Landmark_Detector()
        try:
            _source = int(self.ids.cam.text) 
        except Exception as ee:
            _source = self.ids.cam.text

        self.cap = cv2.VideoCapture( _source )
        self.FaceSwap = face_swap( os.path.join('./portraits', target_face) )
    
    def swap_face(self, *args):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (480,640))
        bboxes, _ = self.face_detector.detect(frame)  # get faces
        if len(bboxes) != 0:
            bbox = bboxes[0] # get the first 
            bbox = bbox.astype(np.int)
            lmks, PRY_3d = self.lmk_detector.detect(frame, bbox)  # get landmarks
            lmks = lmks.astype(np.int)
            frame = self.FaceSwap.run(frame,lmks)
            cv2.imshow("Face Swap", frame)      

    def update(self,*args):    
        Clock.schedule_interval(self.swap_face, self.refresh_dt) 

    def stop(self):
        Clock.unschedule(self.swap_face)
        cv2.destroyWindow('Face Swap')

root_widget = Builder.load_string('''
MyScreenManager:
    PreScreen:
    FrontScreen:

<PreScreen>:
    Image:
        source: ''
        allow_stretch: True
        keep_ratio: False
        size: root.size

        Button:
            text: 'GO'
            font_size:40
            center: root.center
            color: 1,0,1,1
            background_color: 0,0,0,0
            on_release: app.root.current = 'front'

<FrontScreen>:
    name: 'front'
    Image:
        source: ''
        allow_stretch: True
        keep_ratio: False
        size: root.size

        Button:
            id: face_selection
            center: root.center
            text: 'Select a face'
            size: 0.25*root.width, root.height//13
            # on_press: print(root.portraits)

        Label:
            text: 'Camera'
            color: (1, 0.6, 0, 1)
            font_size: 24
            center: 0.2*root.width , 0.65*root.height

        TextInput:
            id: cam
            text: '0'
            font_size: 12
            multiline: False
            center: 0.52*root.width , 0.625*root.height 
            size: (0.3*root.width, root.height//16)
            padding: [0.02*root.width,self.height // 2 - (self.line_height//2) * len(self._lines), 0, 0]
            font_size: dp(18)
            color:(0.11, 0.17, 0.44, 1.0)

        Button:
            id: start
            text: 'START'
            center: root.width//2, 0.3*root.height
            height: root.height//13
            on_release:  root.initialize(face_selection.text)   
            on_release: root.update()  

        Button:
            id: reset
            text: 'RESET'
            center: 1.5*root.width//2, 0.47*root.height
            height: root.height//13
            on_release: root.stop()
''')

class faceApp(App):
    def build(self):
        self.title = 'Face Swap'
        return root_widget

    
faceApp().run()