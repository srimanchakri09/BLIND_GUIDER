from gtts import gTTS
import cv2
import tensorflow as tf
import os
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import time
import numpy as np
label=''
options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.5,
    'gpu': 1.0
}
tfnet=TFNet(options)
cap=cv2.VideoCapture(0);

colors=[tuple(255*np.random.rand(3)) for i in range(5)]

while (cap):
    stime=time.time()
    ret,frame=cap.read()
    results=tfnet.return_predict(frame)
    if ret:
        for color,result in zip(colors,results):
            tl=(result['topleft']['x'],result['topleft']['y'])
            br=(result['bottomright']['x'],result['bottomright']['y'])
            label=result['label']
            frame=cv2.rectangle(frame,tl,br,color,7)
            frame=cv2.putText(frame,label,tl,cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

            '''temp_text = 'this is a '
            my_text = temp_text + label
            lang = "en"
            output = gTTS(text=my_text, lang=lang, slow=False)
            output.save("output.mp3")
            os.system("start output.mp3")'''
        cv2.imshow('frame',frame )
        temp_text = 'this is a '
        my_text = temp_text + label
        lang = "en"
        output = gTTS(text=my_text, lang=lang, slow=False)
        output.save("output.mp3")
        os.system("start output.mp3")



        print('FPS{:.1f}'.format(1/(time.time()-stime)))

        if cv2.waitKey(1) & 0xFF== ord('q'):
            break

    else:
        cap.release()
        cv2.destroyAllWindows()
        break


