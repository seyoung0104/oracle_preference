import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ==============================
# 설정 부분
# ==============================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # 지금 쓰던 기본 모델
DATA_PATH = r"C:\Users\jeong\Downloads\statements\statements\Data\sft_train.jsonl"  # 방금 만든 학습 데이터
OUTPUT_DIR = "models/qwen2.5-0.5B-sft-policy"  # 학습된 모델이 저장될 폴더

MAX_LENGTH = 256  # 프롬프트+답 전체 토큰 길이 최대값
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR = 5e-5


def main():
    print("[INFO] Loading dataset...")
    dataset = load_dataset("json", data_files={"train": DATA_PATH})

    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        trust_remote_code=True,
    )

    # Qwen 계열은 pad_token이 없는 경우가 많아서 eos로 대체
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model.to(device)

    # ==============================
    # 데이터 전처리: 프롬프트 + 답 → 입력/레이블
    # ==============================

    def preprocess(example):
        prompt = example["prompt"]
        answer = example["answer"].strip()

        # 모델에게 보여줄 전체 텍스트
        # 예: "<prompt>\nAnswer: raise"
        full_text = prompt + "\nAnswer: " + answer

        # 프롬프트 + "Answer: " 부분과 전체를 따로 토크나이즈해서
        # 어디부터 답인지 경계선을 잡는다
        prefix_text = prompt + "\nAnswer: "
        prefix_ids = tokenizer(
            prefix_text,
            add_special_tokens=False,
        )["input_ids"]

        enc = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        # 프롬프트 부분은 loss를 계산하지 않도록 -100으로 마스킹
        prefix_len = min(len(prefix_ids), MAX_LENGTH)
        labels[:prefix_len] = [-100] * prefix_len

        enc["labels"] = labels
        return enc

    print("[INFO] Tokenizing dataset...")
    tokenized = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ==============================
    # Trainer 설정
    # ==============================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        bf16=torch.cuda.is_available(),  # GPU 있으면 bf16 사용
        gradient_accumulation_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("[INFO] Start training SFT model...")
    trainer.train()

    print(f"[INFO] Saving model to {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[DONE] SFT training finished.")


if __name__ == "__main__":
    main()
