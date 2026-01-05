python ablation_physisensevlr.py \
  --images_dir /path/to/test/images \
  --gt_csv /path/to/image_description.csv \
  --yolo_weights /path/to/best.pt \
  --configs full no_yolo yolo_only \
  --max_samples 300
