# Waste Detection using Faster R-CNN

Detects waste items in photos, draws a box around each item, and labels its type (Plastic, Paper, Metal, Glass, Foam, Cigarette, Other).

This is **object detection**, not classification: the model says *where* each item is, not just *whether* waste is present.

## Dataset

- **TACO Trash Dataset** (Kaggle: `kneroma/tacotrashdataset`), COCO format
- 1500 images, split 80/20 into 1200 train and 300 validation
- The original 60 classes were **merged into 7 groups**, because many classes had very few images and looked alike

| Group | Examples |
|---|---|
| Plastic | bottles, bags, wrappers, film, cups, lids |
| Paper | cartons, paper cups, paper bags, pizza boxes |
| Metal | cans, foil, bottle caps, pop tabs |
| Glass | bottles, jars, broken glass |
| Foam | foam cups, foam containers, styrofoam |
| Cigarette | cigarette butts |
| Other | batteries, shoes, food waste, rope |

## Architecture

![Faster R-CNN architecture](images/architecture.svg)

Text version:

```
Image -> Augmentation -> ResNet-50 + FPN backbone (features)
      -> RPN (suggests boxes) -> RoI Align -> Box head (class + box refinement) -> NMS -> Final detections
```

1. **Backbone (ResNet-50 + FPN):** turns the image into feature maps (the "eyes").
2. **RPN:** suggests rough boxes where objects might be (the "scout").
3. **RoI Align:** crops each suggested box from the shared feature maps.
4. **Box head:** decides the class of each box and tightens it (the "expert").
5. **NMS:** removes duplicate boxes.

**Training setup:** transfer learning from COCO-pretrained `fasterrcnn_resnet50_fpn_v2` (PyTorch / torchvision), last layer replaced for the 7 groups. SGD (lr 0.005, momentum 0.9), 10 epochs, mixed precision on GPU.

**Augmentation:** horizontal flip, vertical flip, brightness, contrast.

## Results

Metric: **mAP@0.5** on the validation set (a box counts as correct if IoU >= 0.5 and the class is right).

| Version | Classes | Best val mAP@0.5 |
|---|---|---|
| Baseline (v1) | 60 | 0.145 |
| Improved (v2) | 7 | **0.268** |

**Improvements made from v1 to v2:**
- Merged 60 classes into 7 groups
- Applied the EXIF rotation tag so boxes match the photos
- Switched to stronger COCO-pretrained weights (`fasterrcnn_resnet50_fpn_v2`)
- Faster training: smaller input size (640) and mixed precision

> Note: 7 classes is an easier task than 60, so part of the improvement comes from the simpler class setup. The 7-group version is also easier to use in practice.

## How to run

1. Open `waste_detection_faster_rcnn.ipynb` in Google Colab.
2. Set **Runtime -> Change runtime type -> GPU**.
3. If Kaggle asks for login, add your token in the download cell (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
4. Run all cells from top to bottom. The dataset is downloaded automatically.

The notebook trains the model, plots the loss and mAP curves, and shows predictions on validation images. The best weights are saved as `outputs/best.pth` (not included in this repo because of file size).

## Limitations and future work

- Small objects (such as cigarettes and bottle caps) are still detected poorly.
- The "Other" group mixes unrelated items.
- Possible next steps: larger input size, more epochs, per-class score analysis, comparison with YOLO.

## Tech stack

Python, PyTorch, torchvision, pycocotools, PIL, Matplotlib, Google Colab
