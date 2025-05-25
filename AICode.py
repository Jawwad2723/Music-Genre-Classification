from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict
import operator
import os
import pickle
import random
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0*correct/len(testSet)

directory = r"C:\Users\jawwa\OneDrive\Desktop\AICEPAJA\New folder (4)\music.dataset"
f = open("my.dat", 'wb')
i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break   
    for file in os.listdir(os.path.join(directory, folder)):
        file_path = os.path.join(directory, folder, file)
        (rate, sig) = wav.read(file_path)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)

f.close()

dataset = []
def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  

    for x in range(len(dataset)):
        if random.random() <split :      
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])  

trainingSet = []
testSet = []
loadDataset("my.dat" , 0.66, trainingSet, testSet)

leng = len(testSet)
predictions = []
for x in range (leng):
    predictions.append(nearestClass(getNeighbors(trainingSet ,testSet[x] , 5))) 

accuracy1 = getAccuracy(testSet , predictions)
print(f"Genre music classification AI Model accuracy: {accuracy1:.2%}")


dataset = []
def loadDataset(filename):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

loadDataset("my.dat")


results=defaultdict(int)

i=1
for folder in os.listdir(r"C:\Users\jawwa\OneDrive\Desktop\AICEPAJA\New folder (4)\music.dataset"):
    results[i]=folder
    i+=1

(rate,sig)=wav.read(r"C:\Users\jawwa\OneDrive\Desktop\AICEPAJA\New folder (4)\Test.dataset\test_audio.wav"
)
mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature=(mean_matrix,covariance,0)

pred=nearestClass(getNeighbors(dataset ,feature , 5))
# Define a class for the GUI
class MusicGenreClassifierGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Music Genre Classifier")

        # Open and convert JPEG image using Pillow
        pil_image = Image.open(r"C:\Users\jawwa\OneDrive\Desktop\AICEPAJA\backgroud.jpeg")
        self.background_image = ImageTk.PhotoImage(pil_image)
        self.background_label = tk.Label(self.master, image=self.background_image)
        self.background_label.place(relwidth=1, relheight=1)

        # Load Audio File button
        self.load_button = tk.Button(self.master, text="Load Audio File", command=self.load_audio_file, bg="lightblue", font=("Arial", 12))
        self.load_button.pack(pady=50)

        # Classify Genre button
        self.classify_button = tk.Button(self.master, text="Classify Genre", command=self.classify_genre, bg="lightgreen", font=("Arial", 12))
        self.classify_button.pack(pady=10)

        # Accuracy label
        self.accuracy_label = tk.Label(self.master, text="Accuracy: N/A", bg="white", font=("Arial", 12))
        self.accuracy_label.pack()

        # Result label
        self.result_label = tk.Label(self.master, text="Classification Result: N/A", bg="white", font=("Arial", 12))
        self.result_label.pack()

        self.audio_file_path = None

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("WAV files", "*.wav")])
        if file_path:
            print(f"Loaded: {file_path}")
            self.audio_file_path = file_path

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("WAV files", "*.wav")])
        if file_path:
            print(f"Loaded: {file_path}")
            self.audio_file_path = file_path

    def classify_genre(self):
        if self.audio_file_path:
            rate, sig = wav.read(self.audio_file_path)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, 0)  # Assuming label 0 for the test audio

            pred = nearestClass(getNeighbors(dataset, feature, 5))

            results = defaultdict(int)
            i = 1
            for folder in os.listdir(r"C:\Users\jawwa\OneDrive\Desktop\AICEPAJA\New folder (4)\music.dataset"):
                results[i] = folder
                i += 1

            accuracy = getAccuracy(testSet, predictions)
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")
            self.result_label.config(text=f"Classification Result: {results[pred]}")
        else:
            print("Please load an audio file first.")

# Create an instance of Tkinter and the GUI
root = tk.Tk()
app = MusicGenreClassifierGUI(root)
root.mainloop()
