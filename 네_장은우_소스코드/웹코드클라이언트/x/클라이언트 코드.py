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


a=1
clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('192.168.0.2', 8080))
#192.168.84.15
print('연결됐구려 ')

while 1:


    filename = str(a) + ".wav"
    clientSock.sendall(filename.encode('utf-8'))

    data = clientSock.recv(1024)
    data_transferred = 0

    if not data:
        print('파일 %s 가 서버에 없구려' % filename)
        sys.exit()

    nowdir = os.getcwd()
##    print("aaaaaaaaaa",len(data))
    data_received = b''
    while True:
        if len(data) !=1024:
##            print(data)
            data = clientSock.recv(len(data))
        else:
            data = clientSock.recv(1024)
        data_received += data
        
        if not data or data == b'\x08\x00\x08\x00\x07\x00\x07\x00\x07\x00\x05\x00\x07\x00\x07\x00\x06\x00\x08\x00\x07\x00\x05\x00\x07\x00':
            break

    with open(nowdir + "\\" + filename, 'wb') as f:
        f.write(data_received) 
    print('파일 %s 받기 완료. 전송량 %d' % (filename, data_transferred))
    a = a + 1


##    r = sr.Recognizer()
##    try:
##        sample_wav, rate = librosa.core.load(str(a)+".wav")
##    except FileNotFoundError:
##        print("Error: Audio file not found")
##
##    # Recognize speech using Google Speech Recognition
##    korean_audio = sr.AudioFile(str(a)+".wav")
##    print("여기까지")
##    t = ""
##    with korean_audio as source:
##        try:
##            audio = r.record(source)
##            t = r.recognize_google(audio_data=audio, language='ko-KR')
##            print("STT 결과", t)
##            if "토끼" in t :
##                b=t.index("토끼")
##                print("1")
##                shutil.move(str(a)+".wav",str(t[b-7:b+7])+".wav")
##                time.sleep(0.3)
##                shutil.move(str(t[b-7:b+7])+".wav",'static')
##                continue
##            elif "거북이"in t:
##                c=t.index("거북이")
##                shutil.move(str(a)+".wav",str(t[c-7:c+7])+".wav")
##                time.sleep(0.3)
##                shutil.move(str(t[c-7:c+7])+".wav",'static')
####                else:
####                    print("2")
####                    time.sleep(0.3)
####                    shutil.move(str(a)+".wav",'x')  
##            
##            print("Google Speechx` Recognition thinks you said: " + t)
##        except sr.UnknownValueError:
##            # 음성 인식 결과를 얻을 수 없을 때 처리할 내용
##            t = None 
##
##    a=a+1

