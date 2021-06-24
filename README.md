# Real Time FaceSwap with ONNX 
 Using saved onnx models to perform arbitrary face swap task in real time

 ## Major Dependencies 
 ```
onnxruntime==1.6.0
opencv-python==4.4.0
kivy==2.0.0
```

## How does the code work
The code does face detection --> face landmark detection ---> swap face by patching corresponding triangles from face A to face B with delaunay triangulation.
The onnx models used in the first two steps can be found in this [github repo](https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch), one tutorial of the last step can be found [here](https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/). 

The code uses kivy for the simple GUI construction.

## How to play with it

1.Download pictures of whom you would like to swap your with to the `\portrait` folder. A frontal view protrait picture will be ideal for final performance.   

2.Execute `faceswap_kivy.py` to start the GUI.  
 
   * Depending on your machine's acutal hardware setting, change the Camera index to use the right camera.
   * The dropdown menu will contain source portrait you downloaded in the `\portrait` folder. Select one and hit `Start`, you are good to go.
   * If multiple images are in the folder and you would like to use a different face, just clike `reset` and then re-select the one you want to use.
   * With a virtual camera software such as OBS Studio, you can make yourself appear as a different person on your online meetings/classes.
      
3. You may also consider using tools such as `Pyinstaller` to package it to executable and deploy it to other machines without python installed.



https://user-images.githubusercontent.com/23252023/123335848-a932cc80-d50a-11eb-863a-c39f66de87f5.mp4


