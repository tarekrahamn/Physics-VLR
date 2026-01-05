

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore


from ultralytics import YOLO

try:
    import google.generativeai as genai
except Exception:
    genai = None

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def safe_open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def crop_image(img: Image.Image, xyxy: Tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img.width, int(round(x2)))
    y2 = min(img.height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

def list_images(images_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]
    return sorted(files)




def load_gt_csv(gt_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    colmap = {c.lower().strip(): c for c in df.columns}

    if "filename" not in colmap and "image" not in colmap:
        raise ValueError(f"CSV must contain 'filename' column. Found: {list(df.columns)}")

    if "reasoning" not in colmap:
        raise ValueError(f"CSV must contain 'Reasoning' column. Found: {list(df.columns)}")

    filename_col = colmap.get("filename", colmap.get("image"))
    reasoning_col = colmap["reasoning"]
    response_col = colmap.get("response", None)

    out = pd.DataFrame()
    out["filename"] = df[filename_col].astype(str).str.strip()
    out["gt_reasoning"] = df[reasoning_col].apply(normalize_text)

    if response_col:
        out["gt_response"] = df[response_col].apply(normalize_text)
    else:
        out["gt_response"] = ""

    out = out[out["filename"].str.len() > 0].copy()
    return out


# Gemini Reasoner

class GeminiReasoner:
    def __init__(self, api_key: str, model_name: str):
        if genai is None:
            raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, img: Image.Image, instrument_hint: Optional[str] = None) -> str:
        hint = f"Detected instrument label hint: {instrument_hint}\n" if instrument_hint else ""
        prompt = (
            "You are a concise educational assistant for a physics lab.\n"
            f"{hint}"
            "Task:\n"
            "1) Identify the instrument shown.\n"
            "2) Briefly explain its purpose in a physics laboratory.\n"
            "3) Mention 1-2 visual cues that justify your identification.\n"
            "Keep it short (2-4 sentences).\n"
        )
        try:
            resp = self.model.generate_content([prompt, img])
            return normalize_text(getattr(resp, "text", "") or "")
        except Exception as e:
            return normalize_text(f"[Gemini error] {e}")


# YOLO Predict 
def yolo_predict_one(model: YOLO, img_path: str, conf: float, iou: float):
    """
    Returns best_label, best_xyxy, best_conf
    """
    results = model.predict(source=img_path, conf=conf, iou=iou, verbose=False)
    if not results:
        return None, None, None

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None

    confs = r.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    xyxy = r.boxes.xyxy[best_idx].cpu().numpy().tolist()
    xyxy = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

    cls_id = int(r.boxes.cls[best_idx].cpu().numpy())
    names = r.names if hasattr(r, "names") else {}
    best_label = str(names.get(cls_id, cls_id))

    return best_label, xyxy, best_conf


# Metrics

def compute_bleu(refs: List[str], hyps: List[str]) -> float:
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)

def compute_rougeL(refs: List[str], hyps: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for r, h in zip(refs, hyps):
        vals.append(scorer.score(r, h)["rougeL"].fmeasure)
    return float(np.mean(vals)) if vals else 0.0

def compute_bertscore(refs: List[str], hyps: List[str], lang: str) -> float:
    _, _, f1 = bertscore(hyps, refs, lang=lang, rescale_with_baseline=True, verbose=False)
    return float(f1.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder containing test images")
    ap.add_argument("--gt_csv", required=True, help="image_discription.csv path")
    ap.add_argument("--yolo_weights", required=True, help="YOLO best.pt path")
    ap.add_argument("--configs", nargs="+", default=["full", "no_yolo", "yolo_only"],
                    choices=["full", "no_yolo", "yolo_only"])
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all matched samples")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_iou", type=float, default=0.5)
    ap.add_argument("--gemini_model", type=str, default="gemini-1.5-flash")
    ap.add_argument("--bertscore_lang", type=str, default="en")
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    gt_df = load_gt_csv(args.gt_csv)

    img_files = set(list_images(args.images_dir))
    matched = gt_df[gt_df["filename"].isin(img_files)].copy()

    if matched.empty:
        raise RuntimeError(
            "No matching filenames between gt_csv and images_dir.\n"
            "Fix: ensure images_dir contains the exact filenames from CSV 'filename' column."
        )

    if args.max_samples and args.max_samples > 0:
        matched = matched.sample(n=min(args.max_samples, len(matched)), random_state=42)

    matched = matched.reset_index(drop=True)

    yolo = YOLO(args.yolo_weights)

    need_gemini = any(c in ["full", "no_yolo"] for c in args.configs)
    gemini = None
    if need_gemini:
        api_key = os.getenv("GEMINI_API_KEY", "")
        gemini = GeminiReasoner(api_key=api_key, model_name=args.gemini_model)

    rows = []

    for i in tqdm(range(len(matched)), desc="Running Ablation"):
        fn = matched.loc[i, "filename"]
        gt_reasoning = normalize_text(matched.loc[i, "gt_reasoning"])

        img_path = os.path.join(args.images_dir, fn)
        img = safe_open_image(img_path)

        yolo_label, yolo_xyxy, yolo_conf = yolo_predict_one(
            yolo, img_path, conf=args.yolo_conf, iou=args.yolo_iou
        )

        for cfg in args.configs:
            if cfg == "full":
                roi = crop_image(img, yolo_xyxy) if yolo_xyxy is not None else img
                pred_text = gemini.generate(roi, instrument_hint=yolo_label)

            elif cfg == "no_yolo":
                pred_text = gemini.generate(img, instrument_hint=None)

            elif cfg == "yolo_only":
                pred_text = f"Detected instrument: {yolo_label}" if yolo_label else "Detected instrument: [None]"

            rows.append({
                "config": cfg,
                "filename": fn,
                "yolo_label": yolo_label,
                "yolo_conf": yolo_conf,
                "gt_reasoning": gt_reasoning,
                "pred_reasoning": normalize_text(pred_text),
            })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "ablation_outputs.csv")
    out_df.to_csv(out_csv, index=False)

    summary = []
    for cfg in args.configs:
        sub = out_df[out_df["config"] == cfg].copy()

        if cfg == "yolo_only":
            summary.append({
                "config": cfg,
                "BLEU": "",
                "ROUGE_L": "",
                "BERTScore_F1": "",
                "notes": "No free-form reasoning generated"
            })
            continue

        refs = sub["gt_reasoning"].astype(str).tolist()
        hyps = sub["pred_reasoning"].astype(str).tolist()

        bleu = compute_bleu(refs, hyps)
        rougeL = compute_rougeL(refs, hyps)
        bertf1 = compute_bertscore(refs, hyps, lang=args.bertscore_lang)

        summary.append({
            "config": cfg,
            "BLEU": round(bleu, 2),
            "ROUGE_L": round(rougeL, 4),
            "BERTScore_F1": round(bertf1, 4),
            "notes": ""
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(args.out_dir, "ablation_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n✅ Saved:")
    print(" -", out_csv)
    print(" -", summary_csv)
    print("\n📊 Metric Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
