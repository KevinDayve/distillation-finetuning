import os
import torch
from functools import partial
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from dataset_builder import buildDataset
from load_model import Model
from util import collate_fn
import subprocess
import wandb
from gemini_annot import evaluate

# wandbAuth = os.getenv("WANDB_AUTH")
# if wandbAuth is None:
#     raise ValueError("WandB authentication token is not found. Please generate one from https://wandb.ai/authorize and set it as an environment variable named 'WANDB_AUTH'.")

#Config
DataDir = "" #Path to the data directory
annotDir = "" #Path to the annotated directory for evaluation. This is expected to have the structure: root/ActivityName{i}/VideoFile{i}.mp4
OutputDir = "./Gemma3-4b-it-Finetuned"
N = 16 #Frames to extract per video.
BSZ = 4 #Batch size
SYSMSG = "You are an expert video classification model trained to understand human action through careful and diligent analysis."
modelID = 'google/gemma-3-4b-it' #Model ID
activty = []   #list of activities.
prompt = f"Please classify the activity in the video. The activity is one of the following: {activty}."

loraConfig = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
#Load the Gemma 3 model.
model = Model(model_id=modelID, lora_config=loraConfig, load_in_8bit=True, device_map="auto", torch_dtype="bf16")
model, processor = model.loadModel()

#Evaluate the Gemini propreitary model and see if its fit for generating labels for finetuning, if not, we dont proceed.
if os.path.isdir(annotDir) and len(annotDir) > 0:
    metrics = evaluate(
        videoDir=annotDir,
        modelID=modelID,
        sampleSize=100,
    )
    if metrics['accuracy'] < 0.7:
        raise ValueError(
            f"{modelID} is not a good model for generating labels for finetuning. Please try enhancing your prompts or use a different model"
        )


#build the dataset.
dataset = buildDataset(
    datasetPath=DataDir,
    systemMessage=SYSMSG,
    promptTemplate=prompt,
    modelID="gemini-2.0-flash-lite",
    numFrames=N,
    labelsKnown=False,
)

#Set the training config and arguments.
sftConfig = SFTConfig(
    output_dir=OutputDir,
    num_train_epochs=1,
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=BSZ,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim='adamw_torch_fused',
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    logging_steps=10,
    save_strategy='steps',
    save_steps=20,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=False,
    report_to="wandb",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    eval_strategy="no",  # No evaluation during training - as we donn't have a validation set.
    remove_unused_columns=False,
)


sftConfig.remove_unused_columns = False
#For tracking the experiment on wandb.
def checkWandB():
    try:
        loggedIn = subprocess.check_output(["wandb", "whoami"], stderr=subprocess.DEVNULL).decode().strip()
        if "not logged in" in loggedIn.lower():
            print('You are not logged into WandB.')
            print("Run `wandb login` to log in and then pass the authentication token.")
    except Exception:
        print("WandB could not verify your login status. Please check your internet connection and try again.")
        print('You can also run `wandb login` to log in and then pass the authentication token.')

checkWandB()

wandb.init(
    project = "Gemma3-4b-it-Finetuned",
    name = "Gemma3-4b-it-Finetuned",
    config = sftConfig
)
wrappedCollator = partial(collate_fn, processor=processor)
trainer = SFTTrainer(
    model=model,
    args=sftConfig,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=wrappedCollator,
    peft_config=loraConfig,
)

trainer.train()
trainer.save_model(OutputDir)

