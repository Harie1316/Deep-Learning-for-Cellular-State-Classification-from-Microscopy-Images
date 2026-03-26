# Deep Learning for Cellular State Classification from Microscopy Images

A PhD-standard deep learning pipeline for automated subcellular localisation
classification from fluorescence microscopy images.

## Project Overview

| Item                | Detail                                              |
|---------------------|-----------------------------------------------------|
| Task                | 6-class image classification                        |
| Dataset             | Synthetic fluorescence microscopy (1,800 images)   |
| Model 1             | CellNet — custom CNN with residual SE blocks        |
| Model 2             | CBAM-ResNet50 — attention-augmented transfer model  |
| Test accuracy       | 99.63% (both models)                                |
| ROC-AUC (macro)     | 1.0000 (both models)                                |
| Explainability      | Grad-CAM and Grad-CAM++                             |
| Statistical test    | McNemar's test (p = 1.0, models equivalent)         |

## Classes

| Index | Class                  | Morphological Signature                        |
|-------|------------------------|------------------------------------------------|
| 0     | Nucleoplasm            | Diffuse uniform nuclear fill                   |
| 1     | Nuclear membrane       | Ring-localised nuclear boundary                |
| 2     | Cytosol                | Annular cytoplasmic halo                       |
| 3     | Plasma membrane        | Thin outermost cell boundary                   |
| 4     | Mitochondria           | Punctate elongated organelles                  |
| 5     | Endoplasmic reticulum  | Perinuclear reticular network                  |

## Repository Structure
```
cellular_state_classification/
│
├── README.md
├── requirements.txt
├── discussion_and_conclusions.md
│
├── figures/
│   ├── fig1_class_distribution.png
│   ├── fig2_image_montage.png
│   ├── fig3_intensity_distributions.png
│   ├── fig4_mean_images.png
│   ├── fig5_augmentation_pipeline.png
│   ├── fig6_cellnet_training.png
│   ├── fig7_cbam_resnet_training.png
│   ├── fig8_confusion_matrices.png
│   ├── fig9_roc_curves.png
│   ├── fig10_model_comparison.png
│   ├── fig11_gradcam_comparison.png
│   ├── fig12_raw_attribution_maps.png
│   └── fig13_statistical_comparison.png
│
└── checkpoints/
    ├── cellnet_best.pt
    └── cbam_resnet50_best.pt
```

## Reproducing the Pipeline

1. Open the notebook in Google Colab with a GPU runtime (T4 or higher).
2. Run cells 1–15 sequentially.
3. All outputs are regenerated deterministically (SEED = 42).

## Key Design Decisions

- **Label smoothing** (ε = 0.10) prevents overconfident softmax outputs.
- **Cosine annealing with warm restarts** (T₀ = 10) explores the loss landscape.
- **Two-phase training** for CBAM-ResNet50 prevents head initialisation from
  corrupting pretrained backbone weights (Raghu et al., 2019).
- **Weighted random sampling** ensures balanced mini-batches.
- **Mixed precision training** (fp16 + GradScaler) reduces VRAM and accelerates training.
- **Grad-CAM++** over standard Grad-CAM for sharper spatial attribution maps.

## References

See `discussion_and_conclusions.md` for full reference list.

## Citation

If you use this pipeline, please cite the key methodological references:
- He et al. (2016) — Residual networks
- Hu et al. (2018) — Squeeze-and-Excitation networks
- Woo et al. (2018) — CBAM
- Selvaraju et al. (2017) — Grad-CAM
- Loshchilov & Hutter (2017, 2019) — SGDR, AdamW
