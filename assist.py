import speech_recognition as sr
import anthropic
import pyttsx3

# Create a recognizer instance
recognizer = sr.Recognizer()

# Create a text-to-speech engine instance
engine = pyttsx3.init()

# Create an anthropic client instance
client = anthropic.Anthropic(api_key="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")

while True:
    # Use the default system microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        recognizer.dynamic_energy_threshold = 3000

        try:
            audio = recognizer.listen(source, timeout=5.0)
            user_text = recognizer.recognize_google(audio)
            print(user_text)

            # Pass the user's text to the anthropic client
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_text
                            }
                        ]
                    }
                ]
            )

            # Convert the response text to speech
            response_text = message.content[0].text  # Extract the text content
            print(response_text)
            engine.say(response_text)
            engine.runAndWait()

        except sr.UnknownValueError:
            print("Didn't recognize anything.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")