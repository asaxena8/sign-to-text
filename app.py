from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import imutils
from gtts import gTTS
from playsound import playsound
import os
from threading import Thread
main = tkinter.Tk()
main.title("Sign language to speech translation")
main.geometry("1300x1200")
global filename
global classifier
bg = None
playcount = 0
#names = ['Palm','I','Fist','Fist Moved','Thumbs up','Index','OK','Palm Moved','C','Down']
names = ['C','Thumbs Down','Fist','I','Ok','Palm','Thumbs up']
def getID(name):
index = 0
for i in range(len(names)):
if names[i] == name:
index = i
break
return index
bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
49
def deleteDirectory():
filelist = [ f for f in os.listdir('play') if f.endswith(".mp3") ]
for f in filelist:
os.remove(os.path.join('play', f))
def play(playcount,gesture):
class PlayThread(Thread):
def __init__(self,playcount,gesture):
Thread.__init__(self)
self.gesture = gesture
self.playcount = playcount
def run(self):
t1 = gTTS(text=self.gesture, lang='en', slow=False)
t1.save("play/"+str(self.playcount)+".mp3")
playsound("play/"+str(self.playcount)+".mp3")
newthread = PlayThread(playcount,gesture)
newthread.start()
def remove_background(frame):
fgmask = bgModel.apply(frame, learningRate=0)
kernel = np.ones((3, 3), np.uint8)
fgmask = cv2.erode(fgmask, kernel, iterations=1)
res = cv2.bitwise_and(frame, frame, mask=fgmask)
return res
def uploadDataset():
global filename
global labels
labels = []
filename = filedialog.askdirectory(initialdir=".")
pathlabel.config(text=filename)
text.delete('1.0', END)
text.insert(END,filename+" loaded\n\n");
def trainCNN():
global classifier
text.delete('1.0', END)
X_train = np.load('model/X.txt.npy')
Y_train = np.load('model/Y.txt.npy')
text.insert(END,"CNN is training on total images : "+str(len(X_train))+"\n")
if os.path.exists('model/model.json'):
with open('model/model.json', "r") as json_file:
loaded_model_json = json_file.read()
50
classifier = model_from_json(loaded_model_json)
classifier.load_weights("model/model_weights.h5")
classifier._make_predict_function()
print(classifier.summary())
f = open('model/history.pckl', 'rb')
data = pickle.load(f)
f.close()
acc = data['accuracy']
accuracy = acc[9] * 100
text.insert(END,"CNN Hand Gesture Training Model Prediction Accuracy =
"+str(accuracy))
else:
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))
print(classifier.summary())
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =
['accuracy'])
hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True,
verbose=2)
classifier.save_weights('model/model_weights.h5')
model_json = classifier.to_json()
with open("model/model.json", "w") as json_file:
json_file.write(model_json)
f = open('model/history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
f = open('model/history.pckl', 'rb')
data = pickle.load(f)
f.close()
acc = data['accuracy']
accuracy = acc[9] * 100
text.insert(END,"CNN Hand Gesture Training Model Prediction Accuracy =
"+str(accuracy))
def run_avg(image, aWeight):
global bg
if bg is None:
bg = image.copy().astype("float")
return
cv2.accumulateWeighted(image, bg, aWeight)
51
def segment(image, threshold=25):
global bg
diff = cv2.absdiff(bg.astype("uint8"), image)
thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
( cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) == 0:
return
else:
segmented = max(cnts, key=cv2.contourArea)
return (thresholded, segmented)
def webcamPredict():
global playcount
oldresult = 'none'
count = 0
fgbg2 = cv2.createBackgroundSubtractorKNN();
aWeight = 0.5
camera = cv2.VideoCapture(0)
top, right, bottom, left = 10, 350, 325, 690
num_frames = 0
while(True):
(grabbed, frame) = camera.read()
frame = imutils.resize(frame, width=700)
frame = cv2.flip(frame, 1)
clone = frame.copy()
(height, width) = frame.shape[:2]
roi = frame[top:bottom, right:left]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (41, 41), 0)
if num_frames < 30:
run_avg(gray, aWeight)
else:
temp = gray
hand = segment(gray)
if hand is not None:
(thresholded, segmented) = hand
cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
#cv2.imwrite("test.jpg",temp)
#cv2.imshow("Thesholded", temp)
#ret, thresh = cv2.threshold(temp, 150, 255, cv2.THRESH_BINARY +
cv2.THRESH_OTSU)
#thresh = cv2.resize(thresh, (64, 64))
#thresh = np.array(thresh)
#img = np.stack((thresh,)*3, axis=-1)
roi = frame[top:bottom, right:left]
roi = fgbg2.apply(roi);
52
cv2.imwrite("test.jpg",roi)
#cv2.imwrite("newDataset/Fist/"+str(count)+".png",roi)
#count = count + 1
#print(count)
img = cv2.imread("test.jpg")
img = cv2.resize(img, (64, 64))
img = img.reshape(1, 64, 64, 3)
img = np.array(img, dtype='float32')
img /= 255
predict = classifier.predict(img)
value = np.amax(predict)
cl = np.argmax(predict)
result = names[np.argmax(predict)]
if value >= 0.99:
print(str(value)+" "+str(result))
cv2.putText(clone, 'Gesture Recognize as : '+str(result), (10, 25),
cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)
if oldresult != result:
play(playcount,result)
oldresult = result
playcount = playcount + 1
else:
cv2.putText(clone, '', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255,
255), 2)
cv2.imshow("video frame", roi)
cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
num_frames += 1
cv2.imshow("Video Feed", clone)
keypress = cv2.waitKey(1) & 0xFF
if keypress == ord("q"):
break
camera.release()
cv2.destroyAllWindows()
font = ('times', 16, 'bold')
title = Label(main, text='Sign language to speech translation',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)
font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Hand Gesture Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)
53
pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=50,y=150)
markovButton = Button(main, text="Train CNN with Gesture Images",
command=trainCNN)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)
predictButton = Button(main, text="Sign Language Recognition from Webcam",
command=webcamPredict)
predictButton.place(x=50,y=250)
predictButton.config(font=font1)
font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)
deleteDirectory()
main.config(bg='magenta3')
main.mainloop()
# Importing Libraries
import numpy as np
import cv2
import os, sys
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from hunspell import Hunspell
import enchant
54
#import tensorflow.keras as keras
from keras.models import model_from_json
import pyttsx3
engine = pyttsx3.init()
os.environ[&quot;THEANO_FLAGS&quot;] = &quot;device=cuda,
assert_no_cpu_op=True&quot;
#Application :
class Application:
def __init__(self):
self.hs = Hunspell(&#39;en_US&#39;)
self.vs = cv2.VideoCapture(0)
self.current_image = None
self.current_image2 = None
self.json_file = open(&quot;Models\model_new.json&quot;, &quot;r&quot;)
self.model_json = self.json_file.read()
self.json_file.close()
self.loaded_model = model_from_json(self.model_json)
self.loaded_model.load_weights(&quot;Models\model_new.h5&quot;)
self.json_file_dru = open(&quot;Models\model-bw_dru.json&quot; , &quot;r&quot;)
self.model_json_dru = self.json_file_dru.read()
self.json_file_dru.close()
self.loaded_model_dru = model_from_json(self.model_json_dru)
self.loaded_model_dru.load_weights(&quot;Models\model-bw_dru.h5&quot;)
self.json_file_tkdi = open(&quot;Models\model-bw_tkdi.json&quot; , &quot;r&quot;)
self.model_json_tkdi = self.json_file_tkdi.read()
self.json_file_tkdi.close()
55
self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
self.loaded_model_tkdi.load_weights(&quot;Models\model-bw_tkdi.h5&quot;)
Self.jso n_file_smn = open(&quot;Models\model-bw_smn.json&quot; , &quot;r&quot;)
self.model_json_smn = self.json_file_smn.read()
self.json_file_smn.close()
self.loaded_model_smn = model_from_json(self.model_json_smn)
self.loaded_model_smn.load_weights(&quot;Models\model-bw_smn.h5&quot;)
self.ct = {}
self.ct[&#39;blank&#39;] = 0
self.blank_flag = 0
for i in ascii_uppercase:
self.ct[i] = 0
print(&quot;Loaded model from disk&quot;)
self.root = tk.Tk()
self.root.title(&quot;Sign Language To Text Conversion&quot;)
self.root.protocol(&#39;WM_DELETE_WINDOW&#39;, self.destructor)
self.root.geometry(&quot;900x900&quot;)
self.panel = tk.Label(self.root)
self.panel.place(x = 100, y = 10, width = 580, height = 580)
self.panel2 = tk.Label(self.root) # initialize image panel
self.panel2.place(x = 400, y = 65, width = 275, height = 275)
self.T = tk.Label(self.root)
self.T.place(x = 60, y = 5)
self.T.config(text = &quot;Sign Language To Text Conversion&quot;, font =
(&quot;Courier&quot;, 30, &quot;bold&quot;))
self.panel3 = tk.Label(self.root) # Current Symbol
self.panel3.place(x = 500, y = 540)
56
self.T1 = tk.Label(self.root)
self.T1.place(x = 10, y = 540)
self.T1.config(text = &quot;Character :&quot;, font = (&quot;Courier&quot;, 30,
&quot;bold&quot;))
self.panel4 = tk.Label(self.root) # Word
self.panel4.place(x = 220, y = 595)
self.T2 = tk.Label(self.root)
self.T2.place(x = 10,y = 595)
self.T2.config(text = &quot;Word :&quot;, font = (&quot;Courier&quot;, 30,
&quot;bold&quot;))
self.panel5 = tk.Label(self.root) # Sentence
self.panel5.place(x = 350, y = 645)
self.T3 = tk.Label(self.root)
self.T3.place(x = 10, y = 645)
self.T3.config(text = &quot;Sentence :&quot;,font = (&quot;Courier&quot;, 30,
&quot;bold&quot;))
self.T4 = tk.Label(self.root)
self.T4.place(x = 250, y = 690)
self.T4.config(text = &quot;Suggestions :&quot;, fg = &quot;red&quot;, font =
(&quot;Courier&quot;, 30, &quot;bold&quot;))
self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
self.bt1.place(x = 26, y = 745)
self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
self.bt2.place(x = 325, y = 745)
self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
self.bt3.place(x = 625, y = 745)
57
self.str = &quot;&quot;
self.word = &quot; &quot;
self.current_symbol = &quot;Empty&quot;
self.photo = &quot;Empty&quot;
self.video_loop()
def video_loop(self):
ok, frame = self.vs.read()
if ok:
cv2image = cv2.flip(frame, 1)
x1 = int(0.5 * frame.shape[1])
y1 = 10
x2 = frame.shape[1] - 10
y2 = int(0.5 * frame.shape[1])
cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
self.current_image = Image.fromarray(cv2image)
imgtk = ImageTk.PhotoImage(image = self.current_image)
self.panel.imgtk = imgtk
self.panel.config(image = imgtk)
cv2image = cv2image[y1 : y2, x1 : x2]
gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 2)
th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY_INV, 11, 2)
58
ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV +
cv2.THRESH_OTSU)
self.predict(res)
self.current_image2 = Image.fromarray(res)
imgtk = ImageTk.PhotoImage(image = self.current_image2)
self.panel2.imgtk = imgtk
self.panel2.config(image = imgtk)
self.panel3.config(text = self.current_symbol, font = (&quot;Courier&quot;, 30))
self.panel4.config(text = self.word, font = (&quot;Courier&quot;, 30))
self.panel5.config(text = self.str,font = (&quot;Courier&quot;, 30))
predicts = self.hs.suggest(self.word)
#engine.say(self.word)
#engine.runAndWait()
if(len(predicts) &gt; 1):
self.bt1.config(text = predicts[0], font = (&quot;Courier&quot;, 20))
else:
self.bt1.config(text = &quot;&quot;)
if(len(predicts) &gt; 2):
self.bt2.config(text = predicts[1], font = (&quot;Courier&quot;, 20))
else:
self.bt2.config(text = &quot;&quot;)
if(len(predicts) &gt; 3):
self.bt3.config(text = predicts[2], font = (&quot;Courier&quot;, 20))
else:
self.bt3.config(text = &quot;&quot;)
59
self.root.after(5, self.video_loop)
def predict(self, test_image):
test_image = cv2.resize(test_image, (128, 128))
result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
prediction = {}
prediction[&#39;blank&#39;] = result[0][0]
inde = 1
for i in ascii_uppercase:
prediction[i] = result[0][inde]
inde += 1
#LAYER 1
prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
self.current_symbol = prediction[0][0]
engine.say(self.current_symbol)
engine.runAndWait()
#LAYER 2
if(self.current_symbol == &#39;D&#39; or self.current_symbol == &#39;R&#39; or
self.current_symbol == &#39;U&#39;):
prediction = {}
prediction[&#39;D&#39;] = result_dru[0][0]
60
prediction[&#39;R&#39;] = result_dru[0][1]
prediction[&#39;U&#39;] = result_dru[0][2]
prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
self.current_symbol = prediction[0][0]
engine.say(self.current_symbol)
#engine.runAndWait()
if(self.current_symbol == &#39;D&#39; or self.current_symbol == &#39;I&#39; or
self.current_symbol == &#39;K&#39; or
self.current_symbol == &#39;T&#39;):
prediction = {}
prediction[&#39;D&#39;] = result_tkdi[0][0]
prediction[&#39;I&#39;] = result_tkdi[0][1]
prediction[&#39;K&#39;] = result_tkdi[0][2]
prediction[&#39;T&#39;] = result_tkdi[0][3]
prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
self.current_symbol = prediction[0][0]
engine.say(self.current_symbol)
#engine.runAndWait()
if(self.current_symbol == &#39;M&#39; or self.current_symbol == &#39;N&#39; or
self.current_symbol == &#39;S&#39;):
prediction1 = {}
prediction1[&#39;M&#39;] = result_smn[0][0]
prediction1[&#39;N&#39;] = result_smn[0][1]
prediction1[&#39;S&#39;] = result_smn[0][2]
prediction1 = sorted(prediction1.items(), key = operator.itemgetter(1), reverse = True)
61
if(prediction1[0][0] == &#39;S&#39;):
self.current_symbol = prediction1[0][0]
engine.say(self.current_symbol)
#engine.runAndWait()
else:
self.current_symbol = prediction[0][0]
if(self.current_symbol == &#39;blank&#39;):
for i in ascii_uppercase:
self.ct[i] = 0
self.ct[self.current_symbol] += 1
if(self.ct[self.current_symbol] &gt; 60):
for i in ascii_uppercase:
if i == self.current_symbol:
continue
tmp = self.ct[self.current_symbol] - self.ct[i]
if tmp &lt; 0:
tmp *= -1
if tmp &lt;= 20:
self.ct[&#39;blank&#39;] = 0
for i in ascii_uppercase:
self.ct[i] = 0
return
self.ct[&#39;blank&#39;] = 0
for i in ascii_uppercase:
62
self.ct[i] = 0
if self.current_symbol == &#39;blank&#39;:
if self.blank_flag == 0:
self.blank_flag = 1
if len(self.str) &gt; 0:
self.str += &quot; &quot;
self.str += self.word
self.word = &quot;&quot;
else:
if(len(self.str) &gt; 16):
self.str = &quot;&quot;
self.blank_flag = 0
self.word += self.current_symbol
def action1(self):
predicts = self.hs.suggest(self.word)
if(len(predicts) &gt; 0):
self.word = &quot;&quot;
self.str += &quot; &quot;
self.str += predicts[0]
def action2(self):
predicts = self.hs.suggest(self.word)
if(len(predicts) &gt; 1):
self.word = &quot;&quot;
self.str += &quot; &quot;
self.str += predicts[1]
63
def action3(self):
predicts = self.hs.suggest(self.word)
if(len(predicts) &gt; 2):
self.word = &quot;&quot;
self.str += &quot; &quot;
self.str += predicts[2]
def action4(self):
predicts = self.hs.suggest(self.word)
if(len(predicts) &gt; 3):
self.word = &quot;&quot;
self.str += &quot; &quot;
self.str += predicts[3]
def action5(self):
predicts = self.hs.suggest(self.word)
if(len(predicts) &gt; 4):
self.word = &quot;&quot;
self.str += &quot; &quot;
self.str += predicts[4]
def destructor(self):
print(&quot;Closing Application...&quot;)
self.root.destroy()
self.vs.release()
cv2.destroyAllWindows()
print(&quot;Starting Application...&quot;)
(Application()).root.mainloop()
