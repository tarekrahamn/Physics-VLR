
import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

# Metrics
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

# YOLO
from ultralytics import YOLO

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai
except Exception:
    genai = None



def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def crop_image(img: Image.Image, xyxy: Tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img.width, int(round(x2)))
    y2 = min(img.height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return img  # fallback
    return img.crop((x1, y1, x2, y2))

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

@dataclass
class SampleGT:
    filename: str
    reasoning: str
    response: Optional[str] = None


# Load Ground Truth

def load_gt_csv(gt_csv: str) -> Dict[str, SampleGT]:
    df = pd.read_csv(gt_csv)
    cols = {c.lower(): c for c in df.columns}

    # Resolve columns
    filename_col = cols.get("filename") or cols.get("image") or cols.get("img") or None
    reasoning_col = cols.get("reasoning") or cols.get("gt_reasoning") or None
    response_col = cols.get("response") or cols.get("label") or cols.get("instrument") or None

    if filename_col is None or reasoning_col is None:
        raise ValueError(
            f"gt_csv must contain filename/image and reasoning columns. Found columns: {list(df.columns)}"
        )

    gt_map: Dict[str, SampleGT] = {}
    for _, row in df.iterrows():
        fn = str(row[filename_col]).strip()
        rsn = normalize_text(str(row[reasoning_col]))
        resp = normalize_text(str(row[response_col])) if response_col else None
        gt_map[fn] = SampleGT(filename=fn, reasoning=rsn, response=resp)

    return gt_map


# Gemini wrapper

class GeminiReasoner:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        NOTE:
        - Model name can be changed to whichever Gemini model you have access to.
        - In your paper you mention "Gemini 2.5 Flash"; access/model ID may differ.
        """
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_reasoning(self, img: Image.Image, instrument_hint: Optional[str] = None) -> str:
        """
        A prompt aligned with your paper: concise educational assistant,
        identify instrument + purpose + visual cues.
        """
        hint_text = f"Detected instrument label hint: {instrument_hint}.\n" if instrument_hint else ""
        prompt = (
            "You are a concise educational assistant for a physics lab.\n"
            f"{hint_text}"
            "Task:\n"
            "1) Identify the instrument shown.\n"
            "2) Briefly explain its purpose in a physics laboratory.\n"
            "3) Mention 1-2 visual cues that justify your identification.\n"
            "Keep it short (2-4 sentences).\n"
        )
        try:
            resp = self.model.generate_content([prompt, img])
            text = getattr(resp, "text", "") or ""
            return normalize_text(text)
        except Exception as e:
            return normalize_text(f"[Gemini error] {e}")


# YOLO wrapper

def yolo_predict_one(model: YOLO, img_path: str, conf: float = 0.25, iou: float = 0.5):
    """
    Returns:
      - best_label (str or None)
      - best_xyxy (tuple or None)
      - best_conf (float or None)
    """
    results = model.predict(
        source=img_path,
        conf=conf,
        iou=iou,
        verbose=False
    )
    if not results or len(results) == 0:
        return None, None, None

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None

    # pick highest confidence box
    confs = r.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    xyxy = r.boxes.xyxy[best_idx].cpu().numpy().tolist()
    xyxy = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

    cls_id = int(r.boxes.cls[best_idx].cpu().numpy())
    names = r.names if hasattr(r, "names") else {}
    best_label = str(names.get(cls_id, cls_id))

    return best_label, xyxy, best_conf


# ----------------------------
# Metrics
# ----------------------------

def compute_bleu(refs: List[str], hyps: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return float(bleu.score)

def compute_rougeL(refs: List[str], hyps: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r, h in zip(refs, hyps):
        s = scorer.score(r, h)["rougeL"].fmeasure
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0

def compute_bertscore(refs: List[str], hyps: List[str], lang: str = "en") -> float:
    # returns average F1
    P, R, F1 = bertscore(hyps, refs, lang=lang, rescale_with_baseline=True, verbose=False)
    return float(F1.mean().item())

# Main Ablation Runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Directory containing test images")
    ap.add_argument("--gt_csv", required=True, help="CSV with filename + reasoning ground truth")
    ap.add_argument("--yolo_weights", required=True, help="Path to YOLO best.pt")
    ap.add_argument("--configs", nargs="+", default=["full", "no_yolo", "yolo_only"],
                    choices=["full", "no_yolo", "yolo_only"])
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_iou", type=float, default=0.5)
    ap.add_argument("--gemini_model", type=str, default="gemini-1.5-flash")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--bertscore_lang", type=str, default="en")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    gt_map = load_gt_csv(args.gt_csv)

    all_img_files = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    samples = [f for f in all_img_files if f in gt_map]
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]

    if len(samples) == 0:
        raise RuntimeError("No matching samples found between images_dir and gt_csv filenames.")

    # Load YOLO once
    yolo = YOLO(args.yolo_weights)

    # Setup Gemini only if needed
    need_gemini = any(c in ["full", "no_yolo"] for c in args.configs)
    gemini = None
    if need_gemini:
        api_key = os.getenv("GEMINI_API_KEY", "")
        gemini = GeminiReasoner(api_key=api_key, model_name=args.gemini_model)

    rows = []

    for fn in tqdm(samples, desc="Running ablations"):
        img_path = os.path.join(args.images_dir, fn)
        gt = gt_map[fn]
        gt_reason = normalize_text(gt.reasoning)

        # YOLO prediction (used in full + yolo_only; optional hint in full)
        pred_label, pred_xyxy, pred_conf = yolo_predict_one(
            yolo, img_path, conf=args.yolo_conf, iou=args.yolo_iou
        )

        img = safe_open_image(img_path)

        for cfg in args.configs:
            if cfg == "full":
                # YOLO crop -> Gemini
                if pred_xyxy is not None:
                    roi = crop_image(img, pred_xyxy)
                else:
                    roi = img  # fallback if no detection
                pred_text = gemini.generate_reasoning(roi, instrument_hint=pred_label)

            elif cfg == "no_yolo":
                # Gemini on full image (no YOLO crop)
                pred_text = gemini.generate_reasoning(img, instrument_hint=None)

            elif cfg == "yolo_only":
                # No reasoning: we keep a minimal template text so you can still store output,
                # but metrics should be excluded later (or kept as "N/A").
                if pred_label is None:
                    pred_text = "Detected instrument: [None]"
                else:
                    pred_text = f"Detected instrument: {pred_label}"

            else:
                raise ValueError("Unknown config")

            rows.append({
                "config": cfg,
                "filename": fn,
                "yolo_label": pred_label,
                "yolo_conf": pred_conf,
                "gt_reasoning": gt_reason,
                "pred_reasoning": pred_text
            })

    out_csv = os.path.join(args.out_dir, "ablation_outputs.csv")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    # Compute metrics per config (exclude yolo_only from NLG metrics)
    summary_rows = []
    for cfg in args.configs:
        sub = out_df[out_df["config"] == cfg].copy()

        # If you want: skip Gemini-error lines
        # sub = sub[~sub["pred_reasoning"].str.contains(r"\[Gemini error\]", na=False)]

        if cfg == "yolo_only":
            summary_rows.append({
                "config": cfg,
                "BLEU": "",
                "ROUGE_L": "",
                "BERTScore_F1": "",
                "notes": "No free-form reasoning generated"
            })
            continue

        refs = sub["gt_reasoning"].astype(str).apply(normalize_text).tolist()
        hyps = sub["pred_reasoning"].astype(str).apply(normalize_text).tolist()

        bleu = compute_bleu(refs, hyps)
        rougeL = compute_rougeL(refs, hyps)
        bertf1 = compute_bertscore(refs, hyps, lang=args.bertscore_lang)

        summary_rows.append({
            "config": cfg,
            "BLEU": round(bleu, 2),
            "ROUGE_L": round(rougeL, 4),
            "BERTScore_F1": round(bertf1, 4),
            "notes": ""
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "ablation_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved outputs:")
    print(" -", out_csv)
    print(" -", summary_csv)
    print("\nMetric Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
