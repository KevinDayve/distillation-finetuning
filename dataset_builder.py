import os
from typing import List, Any
from datasets import Dataset
from util import frameExtractor, formatResponse
from gemini_annot import uploadVideos, geminiInference

def buildDataset(
        datasetPath: str,
        systemMessage: str,
        promptTemplate: str,
        modelID: str = 'gemini-2.0-flash-lite',
        numFrames: int = 16,
        labelsKnown: bool = False,
) -> Dataset:
    """
    Builds a huggingface dataset from a directory of videos.
    Args:
        datasetPath (str): Root directory containing either:
            - Flat list of video paths.
            - folders of videos, where each folder is a class. (labelsKnown=True)
        systemMessage (str): System message for the model.
        promptTemplate (str): User's prompt for classification.
        modelID (str): Gemini model ID to annotate videos.
        numFrames (int): Number of frames to extract from each video. Default is 16.
        labelsKnown (bool): If True, uses folder names as labels. Default is False.
    Returns:
        Dataset: huggingface dataset with chat-format messages.
    """
    samples = []
    if labelsKnown:
        for label in os.listdir(datasetPath):
            labelPath = os.path.join(datasetPath, label)
            if not os.path.isdir(labelPath):
                continue
            for videopath in os.listdir(labelPath):
                if not videopath.endswith('.mp4'):
                    continue
                video = os.path.join(labelPath, videopath)
                try:
                    frames = frameExtractor(video, numFrames)
                    sample = {
                        'image': frames,
                        'text': promptTemplate,
                        'label': label
                    }
                    samples.append({'messages': formatResponse(systemMessage, sample)})
                except Exception as e:
                    print(f"[WARN]: Skipping {video} due to error: {e}")
    else:
        videopaths = [os.path.join(datasetPath, videopath) for videopath in os.listdir(datasetPath) if videopath.endswith('.mp4')]
        prompts = [promptTemplate] * len(videopaths)
        uploadedVideos = uploadVideos(videopaths)

        labels = geminiInference(uploadedVideos, prompt=prompts, model_id=modelID)
        for video, prompt, label in zip(videopaths, prompts, labels):
            try:
                frames = frameExtractor(video, numFrames)
                sample = {
                    'image': frames,
                    'text': prompt,
                    'label': label
                }
                samples.append({'messages': formatResponse(systemMessage, sample)})
            except Exception as e:
                print(f"[WARN]: Skipping {video} due to error: {e}")
    return Dataset.from_list(samples)