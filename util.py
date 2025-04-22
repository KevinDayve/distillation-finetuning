from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import Batch, AutoProcessor
import torch
from PIL import Image
import cv2
import numpy as np
def formatResponse(
    systemMessages: str,
    sample: Dict[str, Union[str, List[Union[Image.Image, Tuple[Image.Image, float]]]]]) -> List[Dict[str, Any]]:
    """
    Formats the response from the propreitary Gemini model into a Gemma-3 compatible, interleaved format for video comprehension.
    Args:
        systemMessages (str): System messages to be included in the response.
        sample (Dict[str, Union[str, List[Union[Image.Image, Tuple[Image.Image, float]]]]]): The sample response from the Gemini model. Should be of the following form:
            - 'image': List[PIL.Image] or List[Tuple[PIL.Image, float]] for timestamped frames
            - 'text': Prompt string given to the model
            - 'label': Expected answer / Gemini-generated label
    Returns:
        List[Dict[str, Any]]: The formatted response.
    """
    userContent = []
    isTimeStamped = isinstance(sample['image'][0], tuple)
    
    for index, frame in enumerate(sample['image']):
        if isTimeStamped:
            image, timestamp = frame
            userContent.append({"type": "text", "text": f"Frame {index + 1} @ {timestamp:.2f}s:"})
        else:
            image = frame
            userContent.append(
                {"type": "image", "image": image}
            )
        
        #Add final query text.
        userContent.append({"type": "text", "text": sample['text']})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": systemMessages}]
            },
            {
                "role": "user",
                "content": userContent
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample['label']}]
            }
        ]
def frameExtractor(videopath: str, numframes: int = 16) -> List[Image.Image]:
    """
    Extracts a fixed number of frames from a video - evenly spaced.
    If using a Qwen2VL model, you can skip this and use the video directly.
    Args:
        videopath (str): Path to the video file.
        numframes (int): Number of frames to extract.
    Returns:
        List[PIL.Image.Image]: List of extracted frames. (RGB)
    """
    capture = cv2.VideoCapture(videopath)
    N = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if N == 0:
        raise ValueError("No frames found in video.")
    indices = np.linspace(0, N - 1, numframes, dtype=int)
    frames = []

    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = capture.read()
        if ret:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = round(index / FPS, 2)
            frames.append(Image.fromarray(frameRGB), timestamp)
    
    capture.release()
    return frames

def collate_fn(processor: AutoProcessor, examples: List[List[Dict]]) -> Dict[str, Any]:
    """
    Collate function for the Gemma3 model.
    Args:
        processor (AutoProcessor): The processor to use for the Gemma model.
        examples: The formatted responses. They should resemble the following format:
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                }
            ]
    Returns:
        Dict[str, Any]: The collated examples ready for training
    """
    inputs = processor.apply_chat_template(
        examples,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )
    #Mask padding tokens in labels.
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    #mask the image token ID.
    if isinstance(processor, AutoProcessor):
        imageTokens = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.image_token)
    elif isinstance(processor, Qwen2VLProcessor):
        imageTokens = [151652, 151653, 151655]
    else:
        pass

    for imageTokenID in imageTokens:
        labels[labels == imageTokenID] = -100
    
    inputs["labels"] = labels
    return inputs