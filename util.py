from typing import List, Dict, Any, Optional
from transformers import Batch, AutoProcessor
import torch
from PIL import Image
import cv2
import numpy as np
def formatResponse(
    systemMessages: str,
    sample: Dict[str, Any]
) -> List[Dict]:
    """
    Formats the Gemini output into a Gemma3-friendly chat-style conversation.

    Args:
        systemMessages (str): System prompt to guide behaviour.
        sample (Dict): Should include 'image' (List[PIL.Image]), 'text' (prompt), and 'label' (Gemini output).

    Returns:
        List[Dict]: A chat-style conversation compatible with Gemma's processor.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": systemMessages}],
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in sample["image"]],
                {"type": "text", "text": sample["text"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
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

    if N == 0:
        raise ValueError("No frames found in video.")
    indices = np.linspace(0, N - 1, numframes, dtype=int)
    frames = []

    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = capture.read()
        if ret:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frameRGB))
    
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