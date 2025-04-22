from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
# from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedModel
class Model:
    def __init__(self, model_id: str, lora_config: LoraConfig = None, load_in_8bit: bool = True, device_map: str = "auto", torch_dtype: str = "bf16") -> None:
        """
        Initialises the model class with given model ID and quantization settings.
        Args:
            model_id (str): The ID of the model to load. Defaults to None
            lora_config (LoraConfig): The configuration for the LoRA model.
            load_in_8bit (bool): Whether to load the model in 8-bit quantization. Defaults to True.
            device_map (str): The device map to use for loading the model. Defaults to "auto".
            torch_dtype (str): The data type to use for the model. Defaults to "bf16".
        """
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.lora_config = lora_config
        self.torch_dtype = torch_dtype
        if "bf16" in self.torch_dtype:
            self.torch_dtype = torch.bfloat16
        elif "fp16" in self.torch_dtype:
            self.torch_dtype = torch.float16
        elif "fp32" in self.torch_dtype:
            self.torch_dtype = torch.float32
        else:
            raise ValueError("Invalid torch_dtype. Use 'bf16', 'fp16', or 'fp32'.")

    def loadModel(self) -> tuple[PreTrainedModel, AutoProcessor]:
        """
        Loads the model with the given settings.
        Returns:
            PreTrainedModel: The loaded model.
            AutoProcessor: The processor for the model.
        """
        quantisationConfig = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantisationConfig,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
        )
        processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        if self.lora_config is not None:
            model = get_peft_model(model, self.lora_config)
            print(f'LoRA config: {self.lora_config}, Trainable parameters: {model.print_trainable_parameters()}')
            return model, processor
        else:
            return model, processor
        

if __name__ == "__main__":
    #Example
    import os
    import argparse
    from util import frameExtractor, formatResponse
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Video inference using Gemma3 model.")
    parser.add_argument(
        "--video",
        type=str,
        default="obama.mp4",
        help="Path to the video file."
    )
    args = parser.parse_args()

    model = Model(
        model_id="google/gemma-3-4b-it",
        lora_config=None,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype="bf16"
    )
    try:
        model, processor = model.loadModel()
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    if not os.path.exists(args.video) or not args.video.endswith(".mp4"):
        raise FileNotFoundError(f"No file found at {args.video}, please provide a valid video path.")

    # Extract frames and timestamps from the video
    frames = frameExtractor(args.video, numframes=8)
    sample = {
        "image": frames,
        "text": "What is going on here?",
        "label": None
    }

    # Format the response and print it.
    formattedResponse = formatResponse(
        systemMessages='You are an expert evaluator of videos',
        sample=sample
    )
    from pprint import pprint
    pprint(formattedResponse)

    model.eval()
    try:
        inputs = processor.apply_chat_template(
            formattedResponse, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        inputLen = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
            output = output[0][inputLen:]
        decodedOutput = processor.decode(output, skip_special_tokens=True)
        print(decodedOutput)
    except Exception as e:
        raise Exception(f"Error during inference: {e}")