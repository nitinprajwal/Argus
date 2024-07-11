import speech_recognition as sr

# Create a recognizer instance
recognizer = sr.Recognizer()

while True:
    # Use the default system microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        recognizer.dynamic_energy_threshold = 3000

        try:
            audio = recognizer.listen(source, timeout=5.0)
            response = recognizer.recognize_google(audio)
            print(response)

            # Your additional code for handling the recognized text
            # ...

        except sr.UnknownValueError:
            print("Didn't recognize anything.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")