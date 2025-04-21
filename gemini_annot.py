from google import genai
import time
from typing import Union, List, Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")
if apiKey is not None:
    client = genai.Client(api_key=apiKey)
else:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to your Gemini API key.")

def uploadVideos(
        videos: Union[str, List[str]],
        sleep_time: int = 1
) -> Union[genai.types.File, List[genai.types.File]]:
    """
    Uploads videos using Gemini's File API.

    Args:
        videos (str | List[str]): A video file path or list of paths.
        sleep_time (int): Optional wait time between uploads.

    Returns:
        A single genai.types.File or a list of uploaded files.
    """
    if isinstance(videos, str):
        print(f"Uploading: {videos}")
        videofile = client.files.upload(file=videos)
        time.sleep(sleep_time)
        videofile = client.files.get(name=videofile.name)

        if videofile.state.name == "FAILED":
            raise ValueError(f"Upload failed for: {videos}")
        return videofile

    else:
        uploadedFiles = []
        for video in videos:
            print(f"Uploading: {video}")
            fileRef = client.files.upload(file=video)
            time.sleep(sleep_time)

            #Wait for the file to be ACTIve.
            while True:
                fileRef = client.files.get(name=fileRef.name)
                if fileRef.state.name == "ACTIVE":
                    break
                elif fileRef.state.name == "FAILED":
                    raise ValueError(f"Upload failed for: {video}")
                print('.', end='', flush=True)
                time.sleep(1)

            uploadedFiles.append(fileRef)
        print("Videos uploaded successfully.")
        return uploadedFiles

def geminiInference(videofile: Union[genai.Client.files, List[genai.Client.files]], prompt: Union[str, List[str]], model_id: str = 'gemini-2.0-flash-lite') -> str:
    """
    Performs inference on a video file using the Gemini API.
    Args:
        videofile (genai.Client.files): The uploaded video file object or a list of uploaded video file objects.
        prompt (str): The prompt for the inference task.
        model_id (str): The model ID to use for the inference.
    Returns:
        str: The result of the inference.
    """
    if isinstance(videofile, list):
        response = [
            client.models.generate_content(
                model=model_id,
                contents=[
                    video,
                    prompt
                ]
            )
            for video, prompt in zip(videofile, prompt)
        ]
        return [prediction.text for prediction in response]
    else:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                videofile,
                prompt
            ]
        )
        prediction = response.text
        #Check if the prediction is empty or None
        if prediction is None:
            raise ValueError("The prediction is empty or None.")
        return prediction
    

if __name__ == "__main__":
    #Example
    videoFile = 'C:/Users/kevin.dave/Downloads/actor_2.mp4'
    activityList = ['No activity', "Standing", "Walking"]
    prompt = f'What is the activity going on here from the following: {activityList}. Just return the activty name.'
    videofile = uploadVideos(videos=videoFile, sleep_time=1)
    response = geminiInference(videofile=videofile, prompt=prompt)
    print(response)