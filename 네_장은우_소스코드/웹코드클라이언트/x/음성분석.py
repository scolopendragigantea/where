import speech_recognition as sr
import librosa


r = sr.Recognizer()
sample_wav, rate = librosa.core.load(str(3)+".wav")
korean_audio = sr.AudioFile(str(3)+".wav")
with korean_audio as source:
    audio = r.record(source)
t = r.recognize_google(audio_data=audio,language='ko-KR')
print(t)
##if "거북이" in t:
##    print("감지")
