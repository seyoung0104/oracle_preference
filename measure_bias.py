import json
import os
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 설정
# =========================

# ✅ 여기 모델만 바꿔서 쓰면 됨
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct" 
    #llama3-8B-instruct 는 id당 허가 필요
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# testbed / output 경로
TESTBED_PATH = Path("statements/testbed.jsonl")   # testbed 파일
ACTION_MAP_PATH = Path("statements/action_map.json")
OUTPUT_PATH = Path("results/bias_measurements_qwen_realistic.parquet")


# =========================
# 유틸 함수들
# =========================

def load_action_map(tokenizer, action_map_path: Path):
    """
    action_map.json을 읽고, 각 액션별로 '대표 토큰 id 리스트'를 만든다.
    대표 토큰 규칙: multi-token이면 첫 토큰만 사용.
    """
    with open(action_map_path, "r", encoding="utf-8") as f:
        action_map = json.load(f)

    action_token_ids = {}
    for action, word_list in action_map.items():
        ids = []
        for w in word_list:
            # add_special_tokens=False: 순수 토큰들만
            token_ids = tokenizer.encode(w, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            # ✅ D0에서 말한 "대표 토큰 = 첫 토큰" 규칙
            ids.append(token_ids[0])
        if not ids:
            raise ValueError(f"No valid tokens for action '{action}'")
        action_token_ids[action] = ids

    return action_token_ids


def compute_action_logits(last_logits: torch.Tensor, action_token_ids: dict):
    """
    마지막 토큰의 전체 logits에서, 각 action에 해당하는 토큰들의 logit을 모아
    'max logit'을 대표 점수로 사용.
    """
    scores = {}
    for action, ids in action_token_ids.items():
        # ids: 해당 액션의 대표 토큰들 리스트
        token_logits = last_logits[ids]          # shape: (len(ids),)
        scores[action] = float(token_logits.max().item())  # soft margin 전에 쓰는 점수
    return scores


# =========================
# 메인 파이프라인
# =========================

def main():
    print(f"[INFO] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    model.eval()

    # 액션 → 대표 토큰 id 매핑
    print(f"[INFO] Loading action map from {ACTION_MAP_PATH}")
    action_token_ids = load_action_map(tokenizer, ACTION_MAP_PATH)

    rows = []

    print(f"[INFO] Reading testbed from {TESTBED_PATH}")
    with open(TESTBED_PATH, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            scenario = json.loads(line)

            scenario_id = scenario["scenario_id"]
            region = scenario.get("region", "")
            state = scenario.get("state", {})
            templates = scenario["templates"]

            for tpl_idx, tpl in enumerate(templates):
                prompt = tpl.format(**state)

                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    out = model(**inputs)
                    # 마지막 토큰의 pre-softmax logits
                    last_logits = out.logits[0, -1, :].detach().cpu()

                action_logits = compute_action_logits(last_logits, action_token_ids)

                # bias score 예시: I(raise) - I(hold)
                logit_raise = action_logits.get("raise", float("nan"))
                logit_hold = action_logits.get("hold", float("nan"))
                logit_cut = action_logits.get("cut", float("nan"))

                bias_margin = logit_raise - logit_hold  # D0의 margin(Ir - Ih)

                row = {
                    "scenario_id": scenario_id,
                    "region": region,
                    "template_idx": tpl_idx,
                    "model_name": MODEL_NAME,

                    "logit_raise": logit_raise,
                    "logit_hold": logit_hold,
                    "logit_cut": logit_cut,
                    "bias_margin_raise_hold": bias_margin,

                    # state 펼치기
                    "inflation": state.get("inflation", None),
                    "unemployment": state.get("unemployment", None),
                    "growth": state.get("growth", None),
                }

                rows.append(row)

            if line_idx % 50 == 0:
                print(f"[INFO] processed {line_idx} scenarios...")

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving results to {OUTPUT_PATH}")
    # parquet으로 저장 (슬라이드 deliverable 이름과 맞추기)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("[DONE] bias_measurements saved.")


if __name__ == "__main__":
    main()
