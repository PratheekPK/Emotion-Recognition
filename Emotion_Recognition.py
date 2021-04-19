#!/usr/bin/env python
# coding: utf-8



import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        
        result=np.array([])
        
        stft=np.abs(librosa.stft(X))
        
        #extracting the mfccs, chroma, stft and mel from the .wav file
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
        
    return result



def load_ravdess_data():
    feature_list=[]
    emotion_list=[]
    
    #extracting all the files with the .wav format in the folder
    for file in glob.glob("C:\\Users\\hp\\Downloads\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        feature=extract_feature(file)
        feature_list.append(feature)
        emotion_list.append(emotion)
        
    return feature_list, emotion_list



def load_savee_data():
    
    feature_list=[]
    emotion_list=[]
    
    #extracting all the files with the .wav format in the folder
    for file in glob.glob("C:\\Users\\hp\\Downloads\\SAVEE\\ALL\\*.wav"):
        file_name=os.path.basename(file)
        emotion_name=file_name.split("_")[1]
        
        if emotion_name[0:1]=='a':
            emotion=emotions["05"]
            
        elif emotion_name[0:1]=='d':
            emotion=emotions['07']
           
        elif emotion_name[0:1]=='f':
            emotion=emotions['06']
            
        elif emotion_name[0:1]=='h':
            emotion=emotions['03']
            
        elif emotion_name[0:1]=='n':
            emotion=emotions['01']
            
        elif emotion_name[0:2]=='sa':
            emotion=emotions['04']
            
        elif emotion_name[0:2]=='su':
            emotion=emotions['08']
        
        feature=extract_feature(file)
        feature_list.append(feature)
        emotion_list.append(emotion)
        
    return feature_list, emotion_list
    

def load_oaf_tess_data():
    feature_list=[]
    emotion_list=[]
    
    #extracting all the files with the .wav format in the folder
    for file in glob.glob("C:\\Users\\hp\\Downloads\\TESS\\TESS Toronto emotional speech set data\\OAF_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion_name=file_name.split("_")[2]
        #print(emotion_name)
        if emotion_name=='angry.wav':
            emotion=emotions["05"]
            
        elif emotion_name=='disgust.wav':
            emotion=emotions['07']
           
        elif emotion_name=='fear.wav':
            emotion=emotions['06']
            
        elif emotion_name=='happy.wav':
            emotion=emotions['03']
            
        elif emotion_name=='neutral.wav':
            emotion=emotions['01']
            
        elif emotion_name=='sad.wav':
            emotion=emotions['04']
            
        elif emotion_name=='ps.wav':
            emotion=emotions['08']
            
        feature=extract_feature(file)
        feature_list.append(feature)
        emotion_list.append(emotion)
        
    return feature_list, emotion_list

def load_yaf_tess_data():
    feature_list=[]
    emotion_list=[]
    
    #extracting all the files with the .wav format in the folder
    for file in glob.glob("C:\\Users\\hp\\Downloads\\TESS\\TESS Toronto emotional speech set data\\YAF_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion_name=file_name.split("_")[2]
        #print(emotion_name)
        if emotion_name=='angry.wav':
            emotion=emotions["05"]
            
        elif emotion_name=='disgust.wav':
            emotion=emotions['07']
           
        elif emotion_name=='fear.wav':
            emotion=emotions['06']
            
        elif emotion_name=='happy.wav':
            emotion=emotions['03']
            
        elif emotion_name=='neutral.wav':
            emotion=emotions['01']
            
        elif emotion_name=='sad.wav':
            emotion=emotions['04']
            
        elif emotion_name=='ps.wav':
            emotion=emotions['08']
            
        feature=extract_feature(file)
        feature_list.append(feature)
        emotion_list.append(emotion)
        
    return feature_list, emotion_list
        
        


if __name__=='__main__':
    
    emotions={'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'}
    #list of emotions that we will be using
    
    ravdess_feature_list, ravdess_emotion_list=load_ravdess_data()
    #extracting the feature list and emotion list from the ravdess dataset
    
    savee_feature_list, savee_emotion_list=load_savee_data()
    #extracting the feature list and emotion list from the savee dataset
    
    tess_oaf_feature_list, tess_oaf_emotion_list=load_oaf_tess_data()
    #extracting the feature list and emotion list from the tessoaf dataset
    
    tess_yaf_feature_list, tess_yaf_emotion_list=load_yaf_tess_data()
    #extracting the feature list and emotion list from the tessyaf dataset
    
        
    final_feature_list=ravdess_feature_list+savee_feature_list+tess_oaf_feature_list+tess_yaf_feature_list
    #concatenating the feature lists

    final_emotion_list=ravdess_emotion_list+savee_emotion_list+tess_oaf_emotion_list+tess_yaf_emotion_list
    #concatenating the emotion lists

    x_train,x_test,y_train,y_test=train_test_split(np.array(final_feature_list), final_emotion_list, test_size=0.2)
    #train test split

    
    model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    #using the vanilla MLP Classifier present in the sklearn library
    
    model.fit(x_train,y_train)
    
 
    y_pred=model.predict(x_test)
    
  
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    
    print("Accuracy: {:.2f}%".format(accuracy*100))



