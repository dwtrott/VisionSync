import cv2
import time
import os
import openai
import base64
from PIL import Image
from io import BytesIO
from google.cloud import texttospeech
import pygame
import speech_recognition as sr
import numpy as np

# Set OpenAI and Google Cloud API credentials
# NOTE: You must provide valid credentials and keys before using this code.
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "PATH_TO_YOUR_GCLOUD_CREDENTIALS.json"  # Replace with your Google Cloud credentials path

def capture_image(output_path="captured_image.jpg"):
    """
    Captures an image from the webcam and saves it to the specified output path.
    
    Parameters:
        output_path (str): The file path to save the captured image.
        
    Returns:
        str or None: The file path of the captured image if successful, None otherwise.
    """
    camera_index = 0  # Webcam index (e.g., /dev/video0)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open video device /dev/video{camera_index}")
        return None

    # Allow the camera to warm up for a short period
    print("Warming up the camera...")
    time.sleep(2)

    # Try capturing multiple frames to ensure a valid frame is obtained
    valid_frame = False
    frame = None
    for i in range(30):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            valid_frame = True
            frame = test_frame
            print(f"Frame {i + 1} captured successfully.")
        else:
            print(f"Frame {i + 1} failed to capture.")

    # Save the last valid frame if available
    if valid_frame and frame is not None:
        cv2.imwrite(output_path, frame)
        print(f"Image captured and saved to {output_path}")
        cap.release()
        return output_path
    else:
        print("Error: Unable to capture a valid frame.")
        cap.release()
        return None

def resize_and_encode_image(image_path, output_size=(128, 128), rotate_degrees=180):
    """
    Resizes, rotates, and base64-encodes the image at the given path.
    
    Parameters:
        image_path (str): The path to the image file.
        output_size (tuple): The desired output size (width, height).
        rotate_degrees (int): The degrees to rotate the image. Default is 180 degrees.
        
    Returns:
        str: The base64-encoded string of the processed image.
    """
    with Image.open(image_path) as image:
        # Rotate and resize the image
        image = image.rotate(rotate_degrees, expand=True)
        image = image.resize(output_size)

        # Convert image to base64 encoded string
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print("Image resized and encoded successfully.")
        return encoded_image

def analyze_image(encoded_image, user_prompt):
    """
    Sends a user prompt and an image to the OpenAI API and returns the response.
    
    Parameters:
        encoded_image (str): The base64-encoded image data.
        user_prompt (str): The textual prompt or query for the image.
        
    Returns:
        str or None: The response text from the OpenAI API or None if an error occurs.
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ]
        )
        response_text = completion.choices[0].message.content
        print("Response from OpenAI:", response_text)
        return response_text
    except Exception as e:
        print("Error during OpenAI API call:", str(e))
        return None

def text_to_speech(text):
    """
    Converts the given text to speech using Google Cloud Text-to-Speech, saves it as an MP3,
    and then plays the audio.
    
    Parameters:
        text (str): The text content to be synthesized into speech.
    """
    # Initialize the TTS client
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)

    # Specify voice parameters (language and gender)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Specify the audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    # Save the synthesized speech to a file
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'output.mp3'")

    # Play the generated audio file
    play_audio("output.mp3")

def play_audio(file_path):
    """
    Plays an MP3 audio file using pygame with PulseAudio as the audio driver.
    
    Parameters:
        file_path (str): The path to the MP3 file to be played.
    """
    # Set environment variables for SDL to use PulseAudio
    os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
    if "AUDIODEV" in os.environ:
        del os.environ["AUDIODEV"]

    # Initialize pygame mixer and play the audio
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait until the audio playback is finished
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def recognize_command():
    """
    Listens for a specific voice command using the webcam's microphone.
    The intended trigger phrase is "Vision Sync Assist".
    
    Returns:
        str or None: The recognized command in lowercase if successful, None otherwise.
    """
    recognizer = sr.Recognizer()

    # Find the index of the webcam microphone
    mic_list = sr.Microphone.list_microphone_names()
    webcam_mic_index = None
    for i, mic_name in enumerate(mic_list):
        if "C270 HD WEBCAM" in mic_name:  # Adjust to match your webcam mic name
            webcam_mic_index = i
            break

    if webcam_mic_index is None:
        print("Error: Webcam microphone not found.")
        return None

    # Listen for a command from the webcam microphone
    with sr.Microphone(device_index=webcam_mic_index) as source:
        print("Listening for 'Vision Sync Assist'...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
        except sr.WaitTimeoutError:
            print("Listening timed out.")
    return None

def recognize_prompt():
    """
    Announces a request for a user prompt, then listens for the user's prompt and returns it.
    If not understood, it will ask the user to try again.
    
    Returns:
        str or None: The recognized prompt text if successful, None otherwise.
    """
    text_to_speech("Please state request.")
    recognizer = sr.Recognizer()

    # Find the webcam microphone index again
    mic_list = sr.Microphone.list_microphone_names()
    webcam_mic_index = None
    for i, mic_name in enumerate(mic_list):
        if "C270 HD WEBCAM" in mic_name:  # Adjust to match your webcam mic
            webcam_mic_index = i
            break

    if webcam_mic_index is None:
        print("Error: Webcam microphone not found.")
        return None

    # Listen for the user's prompt
    with sr.Microphone(device_index=webcam_mic_index) as source:
        print("Listening for your prompt...")
        try:
            audio = recognizer.listen(source, timeout=15)
            prompt = recognizer.recognize_google(audio)
            print(f"Recognized prompt: {prompt}")
            return prompt
        except sr.UnknownValueError:
            text_to_speech("Sorry, I couldn't understand your prompt. Please try again.")
            print("Sorry, I couldn't understand your prompt.")
        except sr.RequestError as e:
            text_to_speech("Error with the speech recognition service. Please try again.")
            print(f"Error with the speech recognition service: {e}")
        except sr.WaitTimeoutError:
            text_to_speech("Listening for prompt timed out. Please try again.")
            print("Listening for prompt timed out.")
    return None

def main():
    """
    Main entry point for the VisionSync workflow:
    1. Continuously listen for the trigger phrase "Vision Sync Assist".
    2. Upon recognition, prompt the user for a request.
    3. Capture an image via the webcam.
    4. Send the image and request to OpenAI for analysis.
    5. Convert the response text to speech and play it.
    6. If user says "quit", the loop exits.
    """
    while True:
        command = recognize_command()
        if command == "vision sync assist":
            # Recognize the user's prompt
            user_prompt = recognize_prompt()
            if user_prompt:
                # Acknowledge and proceed
                text_to_speech("Taking a picture now.")
                print("Taking a picture as requested...")
                image_path = capture_image()
                if image_path:
                    encoded_image = resize_and_encode_image(image_path)
                    if encoded_image:
                        response_text = analyze_image(encoded_image, user_prompt)
                        if response_text:
                            text_to_speech(response_text)
        elif command == "quit":
            # Gracefully exit the application
            text_to_speech("Goodbye. Exiting now.")
            print("Exiting...")
            break

if __name__ == "__main__":
    main()

