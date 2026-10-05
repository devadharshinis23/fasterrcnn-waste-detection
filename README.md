# Real Waste & Litter Detection with Faster R-CNN (TACO Dataset)

Object detection for litter in the wild — finds and labels **multiple** waste items in a single
photo (bottles, wrappers, cans, cigarettes, ...) using a fine-tuned Faster R-CNN
(ResNet50-FPN-v2) trained on the [TACO](http://tacodataset.org) dataset.

This is the real-detection follow-up to an earlier single-label waste-classification project:
instead of one class per photo, this model localises and classifies every litter item in a scene
with its own bounding box.

## Overview

| | |
|---|---|
| **Task** | Multi-object litter detection (bounding box + class per object) |
| **Model** | Faster R-CNN, ResNet50-FPN-v2 backbone, COCO-pretrained, fine-tuned (43.1M trainable params) |
| **Dataset** | [TACO](https://github.com/pedropro/TACO) — Trash Annotations in Context |
| **Classes** | 10 (top 9 TACO supercategories + "Other Litter" bucket) |
| **Metric** | COCO mAP (AP@[.50:.95], AP@.50), per-class AP |
| **Environment** | Google Colab (GPU runtime) |

## Architecture

```
Photo (any size)
      │
      ▼
Resize + normalize (ImageNet stats)
      │
      ▼
┌─────────────────────────────┐
│  ResNet50 backbone + FPN    │  ← multi-scale feature maps (coarse → fine)
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Region Proposal Network    │  ← proposes ~hundreds of candidate boxes ("something might be here")
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  ROI Align + ROI heads      │  ← per candidate: refine box, classify (10 classes + background)
└─────────────────────────────┘
      │
      ▼
Non-max suppression + score threshold
      │
      ▼
Final detections: [box, class, confidence] × N objects
```

Training minimises four summed losses: RPN objectness, RPN box regression, ROI classification,
ROI box regression — all returned directly by `model(images, targets)` in PyTorch/torchvision.

## Dataset

[TACO](http://tacodataset.org) contains ~1,500 photos of litter (roads, beaches, woods),
annotated in COCO format with 60 fine-grained categories under 28 supercategories. Images are
hosted on Flickr; this project downloads them directly from the dataset's public annotation file
at notebook run-time (no manual download step). In the run below, 1,457 of the 1,500 listed
images downloaded successfully — the rest failed with transient `502 Bad Gateway` errors from
Flickr's servers (re-running the download cell may recover some of these; it skips files that
already exist).

Because most of the 60 fine-grained categories have very few examples, this project follows the
same reduction used in TACO's own paper ("TACO-10"): the 9 most frequent supercategories are kept
as individual classes, and everything else is bucketed into a 10th `Other Litter` class. In the
downloaded subset, the 9 kept classes were: **Plastic bag & wrapper, Cigarette, Unlabeled litter,
Bottle, Bottle cap, Can, Other plastic, Carton, Cup**.

## Repository contents

```
.
├── Real_Waste_Litter_Detection_TACO_FasterRCNN.ipynb   # full pipeline: data → train → eval → demo
└── README.md
```

## Getting started (Google Colab)

1. Open `Real_Waste_Litter_Detection_TACO_FasterRCNN.ipynb` in Colab.
2. Set the runtime type to **GPU** (Runtime → Change runtime type → T4 GPU or better).
3. Run all cells top to bottom.
   - The dataset-download cell fetches ~1,500 images from Flickr; expect roughly 1,450–1,500 to
     succeed, and some transient server errors are normal.
   - Training runs for `NUM_EPOCHS` (default 12); took ~900s/epoch (~15 min) on a Colab T4 GPU in
     this run, ~3 hours total.
4. The notebook saves the best checkpoint to `/content/checkpoints/fasterrcnn_taco_best.pth` and
   prints COCO mAP plus a per-class AP breakdown at the end.

### Requirements

Installed automatically by the notebook's first cell:

```
torch
torchvision >= 0.15   # needed for transforms.v2 / tv_tensors and fasterrcnn_resnet50_fpn_v2
pycocotools
requests
tqdm
matplotlib
```

## Results

From a full 12-epoch run (split: train=1165, val=146, test=146 images). Best checkpoint selected
at **epoch 8** (lowest val_loss=0.2046); train_loss kept falling through epoch 12 (0.265 → 0.102)
while val_loss plateaued and crept back up after epoch 8 — a sign of overfitting past that point.

| Metric | Value |
|---|---|
| AP@[.50:.95] | 0.019 |
| AP@.50 | 0.023 |
| AP@.75 | 0.023 |
| AP (small objects) | 0.000 |
| AP (medium objects) | 0.001 |
| AP (large objects) | 0.026 |
| AR@100 | 0.051 |

Per-class AP@[.50:.95]:

| Class | AP |
|---|---|
| Bottle | 0.090 |
| Other plastic | 0.035 |
| Plastic bag & wrapper | 0.028 |
| Bottle cap | 0.020 |
| Cup | 0.015 |
| Other Litter | 0.002 |
| Cigarette | 0.000 |
| Unlabeled litter | 0.000 |
| Can | 0.000 |
| Carton | 0.000 |

**Honest read of these numbers:** this is a genuinely weak result, not just "small-dataset-so-
expect-modest-numbers" — 4 of 10 classes scored 0 AP, and the overfitting signature above means
the model stopped generalising well before training even finished. This is the real output of
the run, kept here unedited rather than replaced with placeholder/target numbers, and the next
section lists where to start improving it.

## Known limitations

- **Small dataset** (~1,450 images after download) relative to standard detection benchmarks —
  this alone caps how high mAP can realistically go.
- **Class imbalance** persists even after the TACO-10 reduction — Plastic bag & wrapper (824
  annotations) vs. e.g. Can (270) or rarer classes.
- **Flickr-hosted images** mean some links fail intermittently (502 errors seen in this run),
  slightly shrinking the effective dataset on each download.
- **Fixed square resize (512×512)** likely hurts small-object detection — AP on small objects was
  0.000 in this run, consistent with that.
- **Overfitting after epoch 8** (val_loss rising while train_loss kept falling) suggests either
  fewer epochs with earlier stopping, a lower learning rate, or more augmentation would help.

## Possible extensions / next steps

- Train for fewer epochs with stricter early stopping around the val-loss minimum (observed at
  epoch 8 in this run), or lower the learning rate.
- Check RPN anchor box sizes against the dataset's actual object-size distribution — many litter
  items (bottle caps, cigarettes) are small relative to the 512×512 input.
- Ensemble 2–3 Faster R-CNN backbone variants (`resnet50_fpn`, `resnet50_fpn_v2`,
  `mobilenet_v3_large_fpn`) with weighted fusion, matched per-box via IoU across models.
- Deploy `predict_image()` behind a small Streamlit/Gradio app for live photo upload → annotated
  output demos, once detection quality improves.

## References

- Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection*.
  [arXiv:2003.06975](https://arxiv.org/abs/2003.06975)
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object
  Detection with Region Proposal Networks*. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- [TACO dataset & toolkit](https://github.com/pedropro/TACO)
- [torchvision detection models documentation](https://pytorch.org/vision/stable/models.html#object-detection)
