import socket
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
def receive_wav_file(server_ip, server_port, save_path):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        
        with open(save_path, 'wb') as file:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                file.write(data)

        print("파일 수신 완료")
    except Exception as e:
        print("파일 수신 중 오류 발생:", e)
    finally:
        client_socket.close()

def main():
    server_ip = '192.168.0.2'
    server_port = 1234  # 서버와 동일한 포트 번호를 사용해야 합니다.
    a = 0
    while 1:
        a=a+1
        save_path = str(a)+'.wav'
        receive_wav_file(server_ip, server_port, save_path)
        print("파일 다받음")
        r = sr.Recognizer()
        try:
            sample_wav, rate = librosa.core.load(str(a)+".wav")
        except FileNotFoundError:
            print("Error: Audio file not found")

    # Recognize speech using Google Speech Recognition

        korean_audio = sr.AudioFile(str(a)+".wav")
        print("여기까지")
        t = ""
        try:
            with korean_audio as source:
                audio = r.record(source)
                t = r.recognize_google(audio_data=audio, language='ko-KR')
            if "토끼" in t :
                b=t.index("토끼")
                print("1")
                shutil.move(str(a)+".wav",str(t[b-7:b+7])+".wav")
                time.sleep(0.3)
                shutil.move(str(t[b-7:b+7])+".wav",'static')
            elif "거북이"in t:
                c=t.index("거북이")
                shutil.move(str(a)+".wav",str(t[c-7:c+7])+".wav")
                time.sleep(0.3)
                shutil.move(str(t[c-7:c+7])+".wav",'static')
            print("STT 결과:", t)
        except sr.UnknownValueError:
            print("음성을 인식할 수 없습니다.")
        except sr.RequestError:
            print("Google Web API에 접근할 수 없습니다.")


if __name__ == '__main__':
    main()
