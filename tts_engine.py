import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 145) 
    engine.say(text)
    engine.runAndWait()
    engine.stop()