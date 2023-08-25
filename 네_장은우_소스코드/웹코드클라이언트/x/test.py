import speech_recognition as sr
import librosa
import shutil
r = sr.Recognizer()
a = 2
ii=0
k=6
while k:
    try:
        sample_wav, rate = librosa.core.load(str(2)+".wav")
    except FileNotFoundError:
        print("Error: Audio file not found")

    # Recognize speech using Google Speech Recognition
    korean_audio = sr.AudioFile(str(a)+".wav")
    with korean_audio as source:
        try:
            audio = r.record(source)
            t = r.recognize_google(audio_data=audio, language='ko-KR')
            print("Google Speech Recognition thinks you said: " + t)
        except sr.UnknownValueError:
            print("Error: Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Error: Could not request results from Google Speech Recognition service; {0}".format(e))

    if "토끼" in t :
        b=t.index("토끼")
        print("1")
        shutil.move(str(a)+".wav",str(t[b-7:b+7])+".wav")
        print("11")
        shutil.move(str(t[b-7:b+7])+".wav",'static')
        a=a+1
        k=k-1
        continue
    if "거북이"in t:
        c=t.index("거북이")
        shutil.move(str(a)+".wav",str(t[c-7:c+7])+".wav")
        shutil.move(str(t[c-7:c+7])+".wav",'static')
    else:
        print("2")
        shutil.move(str(a)+".wav",'x')
    a=a+1
    k=k-1
