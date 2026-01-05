export GEMINI_API_KEY="YOUR_KEY"
python ablation_Study.py \
  --images_dir "/path/to/test_images" \
  --gt_csv "/path/to/image_discription.csv" \
  --yolo_weights "/path/to/best.pt" \
  --configs full no_yolo yolo_only \
  --max_samples 60
