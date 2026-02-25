# Temporal Completion Pilot Dataset

This directory contains the minimal dataset for the LoRA fine-tuning pilot run.

## Structure
```
temporal_completion_pilot/
├── videos/
│   └── doer_pilot.mp4        # 121-frame drone video clip (not tracked by git — large binary)
└── t5_xxl/
    └── doer_pilot.pickle     # Pre-computed T5-XXL text embeddings (not tracked by git)
```

## How to populate

**Step 1 — Copy the pilot video:**
```bash
cp assets/doer_video.mp4 datasets/temporal_completion_pilot/videos/doer_pilot.mp4
```

**Step 2 — Generate T5 embeddings:**
```bash
python scripts/get_t5_embeddings.py \
  --input_path datasets/temporal_completion_pilot/videos/doer_pilot.mp4 \
  --output_path datasets/temporal_completion_pilot/t5_xxl/doer_pilot.pickle \
  --prompt "aerial drone video flying over a sports field"
```

## Usage
This dataset is used by `scripts/run_temporal_completion_lora_pilot.sh`.
