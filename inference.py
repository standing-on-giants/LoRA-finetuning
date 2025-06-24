import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Example: Using transformers pipeline for VQA (replace with your model as needed)
from transformers import pipeline, ViltProcessor, ViltForQuestionAnswering, \
        AutoProcessor, AutoModelForVision2Seq, \
        BlipProcessor, BlipForQuestionAnswering
#from qwen_vl_utils import process_vision_info
from peft import PeftModel, PeftConfig

adapter_path = "./vilt_vqa_lora_r_16/"  # where adapter_model.safetensors and adapter_config.json are
peft_config = PeftConfig.from_pretrained(adapter_path)

adapter_dir = "./blip_vqa_lora_r_16/"  # where adapter_model.safetensors and adapter_config.json are
peft_config_blip = PeftConfig.from_pretrained(adapter_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    # # parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    # parser.add_argument('--csv_path', type=str, required=True, help='Path to curated dataset CSV')
    args = parser.parse_args()

    #image_dir = "./images/small"
    image_dir = args.image_dir
    csv_path = "./full_data_curated.csv"

    # Load metadata CSV
    df = pd.read_csv(csv_path)

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ViLT model definitions
    processor_vilt = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
    model_vilt.eval()

    # ViLT finetuned model definitions
    base_model__vilt_finetune = ViltForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path)
    model__vilt_finetune = PeftModel.from_pretrained(base_model__vilt_finetune, adapter_path)
    processor__vilt_finetune = ViltProcessor.from_pretrained(peft_config.base_model_name_or_path)
    model__vilt_finetune.eval()
    model__vilt_finetune.to(device)

    # BLIP model definitions
    BLIP_vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base", device=device)

    # BLIP finetuned model definitions
    base_model_blip = BlipForQuestionAnswering.from_pretrained(peft_config_blip.base_model_name_or_path)
    model__blip_finetuned = PeftModel.from_pretrained(base_model_blip, adapter_dir)
    processor__blip_finetuned = BlipProcessor.from_pretrained(peft_config_blip.base_model_name_or_path)
    model__blip_finetuned.eval()
    model__blip_finetuned.to(device)

    # # Qwen model definitions
    # Qwen_model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    # processor_qwen = AutoProcessor.from_pretrained(Qwen_model_id)
    # model_qwen = AutoModelForVision2Seq.from_pretrained(Qwen_model_id, device_map="auto", torch_dtype=torch.float16)
    # # model_qwen = AutoModelForVision2Seq.from_pretrained(
    # #     Qwen_model_id,
    # #     torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    # # ).to(device)

    # Smol model definitions
    processor_smol = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model_smol = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

    # answers from original ViLT model
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{row['image_name']}"
        # print(image_path)
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor_vilt(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_vilt(**encoding)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer = model_vilt.config.id2label[predicted_idx]
        except Exception as e:
            answer = "error"
            print(e)
        
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)
        #print(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results__ViLT_original.csv", index=False)

    # answers from finetuned ViLT model
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor__vilt_finetune(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model__vilt_finetune(**encoding)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer = model__vilt_finetune.config.id2label[idx]
        except Exception as e:
            answer = "error"
        
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)
    
    # answers from original BLIP model
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                prediction = BLIP_vqa_pipeline(image=image, question=question)
                answer = prediction[0]['answer']
        except Exception as e:
            answer = "error"
        
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results_dup.csv", index=False)

    # answers from finetuned BLIP model
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor__vilt_finetune(image, question, return_tensors="pt").to(device)
            inputs = processor__blip_finetuned(images=image, text=question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model__blip_finetuned(**inputs)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer = model__blip_finetuned.config.id2label[predicted_idx]
        except Exception as e:
            answer = "error"
        
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)
    
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

    # # answers from original Qwen model
    # generated_answers = []
    # for idx, row in tqdm(df.iterrows(), total=len(df)):
    #     image_path = f"{image_dir}/{row['image_name']}"
    #     question = str(row['question'])
    #     try:
    #         messages = [
    #             {"role": "user", "content": [
    #                 {"type": "image", "image": image_path},
    #                 {"type": "text", "text": "answer in one word only" + question}
    #             ]}
    #         ]
    #         text = processor_qwen.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )
    #         image_inputs, video_inputs = process_vision_info(messages)

    #         with torch.no_grad():
    #             inputs = processor_qwen(
    #                 text=[text],
    #                 images=image_inputs,
    #                 videos=video_inputs,
    #                 padding=True,
    #                 return_tensors="pt",
    #             )
    #             inputs = inputs.to(device)

    #             generated_ids = model_qwen.generate(**inputs, max_new_tokens=128)
    #             generated_ids_trimmed = [
    #                 out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #             ]
    #             answer = processor_qwen.batch_decode(
    #                 generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #             )[0]
    #     except Exception as e:
    #         answer = "error"
        
    #     # Ensure answer is one word and in English (basic post-processing)
    #     answer = str(answer).split()[0].lower()
    #     generated_answers.append(answer)
    
    # df["generated_answer"] = generated_answers
    # df.to_csv("results__Qwen_original.csv", index=False)

    # answers from original Smol model
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{image_dir}/{row['image_name']}"
        question = str(row['question'])
        prompt = f"<image>\nQuestion: {question}\nAnswer:"
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                prediction = BLIP_vqa_pipeline(image=image, question=question)
                answer = prediction[0]['answer']

                inputs = processor_smol(images=[image], text=[prompt], return_tensors="pt").to(model_smol.device)
                generated_ids = model_smol.generate(**inputs, max_new_tokens=50)
                answer = processor_smol.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            answer = "error"
        
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results__Smol_original.csv", index=False)


if __name__ == "__main__":
    main()