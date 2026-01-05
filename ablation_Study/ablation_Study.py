import os
import re
import argparse
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


# -------------------------
# Helpers
# -------------------------

def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def safe_open_image(path):
    return Image.open(path).convert("RGB")

def crop_image(img, xyxy):
    if xyxy is None:
        return img
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img.width, int(round(x2)))
    y2 = min(img.height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

def find_all_images(root_dir):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    out = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(r, f))
    return sorted(out)

def base_name_clean(x):
    # remove path, strip spaces
    x = str(x).strip()
    x = x.split("/")[-1].split("\\")[-1]
    return x.strip()

def to_lower_ext_name(name):
    # normalize extension case
    return name.strip()

def build_image_index(image_paths):
    """
    Build mapping from lowercase filename -> full path
    so case mismatch doesn't break matching.
    """
    idx = {}
    for p in image_paths:
        fn = os.path.basename(p)
        idx[fn.lower()] = p
    return idx


# -------------------------
# Metrics
# -------------------------

def compute_bleu(refs, hyps):
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)

def compute_rougeL(refs, hyps):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for r, h in zip(refs, hyps):
        vals.append(scorer.score(r, h)["rougeL"].fmeasure)
    return float(np.mean(vals)) if vals else 0.0

def compute_bertscore(refs, hyps, lang="en"):
    _, _, f1 = bertscore(hyps, refs, lang=lang, rescale_with_baseline=True, verbose=False)
    return float(f1.mean().item())


# -------------------------
# Gemini
# -------------------------
class GeminiReasoner:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, img, instrument_hint=None):
        if instrument_hint:
            prompt = f"""
Detected instrument label: {instrument_hint}

You are an educational assistant for a physics laboratory.

Task:
1) Confirm whether the detected label matches the visual appearance.
2) Explain the instrument’s function in a physics laboratory.
3) Mention 1–2 specific visual cues that justify the identification.

Keep the explanation concise (2–4 sentences).
"""
        else:
            prompt = """
You are an educational assistant for a physics laboratory.

Task:
1) Identify the instrument shown in the image.
2) Explain its function in a physics laboratory.
3) Mention 1–2 visual cues that support your identification.

Keep the explanation concise (2–4 sentences).
"""

        try:
            resp = self.model.generate_content([prompt, img])
            return normalize_text(resp.text)
        except Exception as e:
            return f"[Gemini error] {e}"



# -------------------------
# YOLO
# -------------------------

def yolo_predict_one(model, img_path, conf=0.25, iou=0.5):
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


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="/content/test_images/images")
    parser.add_argument("--gt_csv", default="/content/image_discription.csv")
    parser.add_argument("--yolo_weights", default="/content/best.pt")
    parser.add_argument("--configs", nargs="+", default=["full", "no_yolo", "yolo_only"],
                        choices=["full", "no_yolo", "yolo_only"])
    parser.add_argument("--max_samples", type=int, default=302)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.5)
    parser.add_argument("--gemini_model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--out_dir", type=str, default="/content/results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load GT
    gt = pd.read_csv(args.gt_csv)
    if "filename" not in gt.columns or "Reasoning" not in gt.columns:
        raise ValueError(f"CSV must contain columns: filename, Reasoning. Found: {gt.columns.tolist()}")

    gt["filename_clean"] = gt["filename"].apply(base_name_clean)
    gt["gt_reasoning"] = gt["Reasoning"].apply(normalize_text)

    # 2) Find images (walk nested folders too)
    img_paths = find_all_images(args.images_dir)
    if len(img_paths) == 0:
        raise ValueError(f"No images found under {args.images_dir}. Check unzip path/folder name.")

    img_index = build_image_index(img_paths)  # lower(filename) -> full path

    # 3) Match
    gt["filename_lower"] = gt["filename_clean"].str.lower()
    gt["img_path"] = gt["filename_lower"].map(img_index)

    matched = gt[gt["img_path"].notna()].copy()

    # Debug print if mismatch
    print("Total GT rows:", len(gt))
    print("Total image files found:", len(img_paths))
    print("Matched rows:", len(matched))

    if len(matched) == 0:
        # show examples to fix
        print("\n❌ No matches found.")
        print("Example CSV filenames (clean):", gt["filename_clean"].head(10).tolist())
        print("Example image filenames:", [os.path.basename(p) for p in img_paths[:10]])
        raise ValueError("No matching filenames found between CSV and images_dir.")

    # 4) Safe sampling
    N = min(args.max_samples, len(matched)) if args.max_samples > 0 else len(matched)
    matched = matched.sample(n=N, random_state=42).reset_index(drop=True)
    print(f"✅ Using samples: {len(matched)}")

    # 5) Load YOLO
    yolo = YOLO(args.yolo_weights)

    # 6) Gemini setup if needed
    need_gemini = any(c in ["full", "no_yolo"] for c in args.configs)
    gemini = None
    if need_gemini:
        api_key = os.getenv("GEMINI_API_KEY", "")
        gemini = GeminiReasoner(api_key=api_key, model_name=args.gemini_model)

    rows = []

    # 7) Run configs
    for i in tqdm(range(len(matched)), desc="Running Ablation"):
        img_path = matched.loc[i, "img_path"]
        gt_reason = matched.loc[i, "gt_reasoning"]
        fn = matched.loc[i, "filename_clean"]

        img = safe_open_image(img_path)

        # YOLO prediction
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
                "img_path": img_path,
                "yolo_label": yolo_label,
                "yolo_conf": yolo_conf,
                "gt_reasoning": gt_reason,
                "pred_reasoning": normalize_text(pred_text),
            })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "ablation_outputs.csv")
    out_df.to_csv(out_csv, index=False)

    # 8) Metrics summary
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


!python ablation_physisensevlr_full.py
