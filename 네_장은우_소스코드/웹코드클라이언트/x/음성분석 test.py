from socket import *
import os
import sys
import time
import speech_recognition as sr
import time
import queue, os, threading
import urllib.request
import urllib.request as ul
import json
import requests
import json
import shutil
import librosa
for a in range(1,4):
    r = sr.Recognizer()
    try:
        sample_wav, rate = librosa.core.load(str(a)+".wav")
    except FileNotFoundError:
        print("Error: Audio file not found")

# Recognize speech using Google Speech Recognition

    korean_audio = sr.AudioFile(str(a)+".wav")
    print("여기까지")
    t = ""
    with korean_audio as source:
            audio = r.record(source)
            t = r.recognize_google(audio_data=audio, language='ko-KR')
            print("STT 결과", t)
            if "토끼" in t :
                b=t.index("토끼")
                print("1")
                shutil.move(str(a)+".wav",str(t[b-7:b+7])+".wav")
                time.sleep(0.3)
                shutil.move(str(t[b-7:b+7])+".wav",'static')
                continue
            elif "거북이"in t:
                c=t.index("거북이")
                shutil.move(str(a)+".wav",str(t[c-7:c+7])+".wav")
                time.sleep(0.3)
                shutil.move(str(t[c-7:c+7])+".wav",'static')
            else:
                print("2")
                time.sleep(0.3)
                shutil.move(str(a)+".wav",'x')  
