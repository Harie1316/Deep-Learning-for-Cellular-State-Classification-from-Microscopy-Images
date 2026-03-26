
# Deep Learning for Cellular State Classification from Fluorescence Microscopy Images

---

## Abstract

We present a complete deep learning pipeline for automated classification of
subcellular localisation states from fluorescence microscopy images. Two
architecturally distinct models were developed and evaluated: CellNet, a
custom convolutional neural network incorporating residual connections and
Squeeze-and-Excitation (SE) channel recalibration, trained entirely from
scratch; and CBAM-ResNet50, a pretrained ResNet50 backbone augmented with
Convolutional Block Attention Modules (CBAM) and fine-tuned using a
two-phase transfer learning strategy. Both models were trained on a
procedurally generated synthetic fluorescence microscopy dataset comprising
1,800 images across six biologically defined subcellular compartment classes.
Gradient-weighted Class Activation Mapping (Grad-CAM and Grad-CAM++) was
applied to assess the spatial interpretability of model predictions. Both
models achieved 99.63% test accuracy with a macro ROC-AUC of 1.0000.
McNemar's test confirmed statistical equivalence of the two models
(p = 1.0). CBAM-ResNet50 produced consistently lower Grad-CAM++ spatial
entropy across all classes, indicating more spatially precise and biologically
interpretable attribution patterns.

---

## 1. Introduction

Fluorescence microscopy is the primary modality for visualising subcellular
structures in modern cell biology. The automated classification of protein
localisation patterns from microscopy images — a task historically performed
by trained biologists — presents a significant opportunity for deep learning.
Manual annotation is slow, subjective, and cannot scale to the throughput
demanded by high-content screening campaigns in drug discovery and functional
genomics (Ljosa & Carpenter, 2009; Caicedo et al., 2017).

The visual similarity between certain cellular states constitutes the central
challenge: the boundary between cytosolic and plasma membrane localisation,
or between nucleoplasmic and nuclear membrane signal, can be ambiguous even
to domain experts. Convolutional neural networks (CNNs) have demonstrated
strong performance on such tasks (Kraus et al., 2017; Sullivan et al., 2018),
but two persistent concerns limit their deployment in biological research:
first, whether the model is attending to biologically meaningful spatial
features rather than image artefacts; and second, whether complex pretrained
architectures offer any advantage over lightweight custom models when labelled
data is scarce.

This project addresses both concerns through parallel development of a
custom CNN and an attention-augmented transfer learning model, with explicit
explainability analysis using Grad-CAM to validate biological plausibility
of the learned representations.

---

## 2. Dataset and Preprocessing

### 2.1 Synthetic Dataset Generation

Due to access constraints on large-scale proprietary microscopy datasets, a
procedurally generated synthetic dataset was employed. This approach is
methodologically precedented in the computational biology literature
(Ljosa & Carpenter, 2009; Eulenberg et al., 2017) and offers the advantage
of complete ground-truth control — each image is generated with a known,
deterministic morphological signature corresponding to its class label,
eliminating label noise entirely.

Six subcellular localisation classes were simulated using OpenCV-based
procedural generation:

- **Nucleoplasm**: diffuse, uniform fluorescence filling the nuclear volume,
  rendered as filled ellipses with superimposed chromatin texture.
- **Nuclear membrane**: ring-localised signal at the nuclear boundary,
  rendered as thin elliptical outlines with variable thickness.
- **Cytosol**: annular cytoplasmic signal surrounding the nucleus, rendered
  as a large filled ellipse with the nuclear region zeroed out.
- **Plasma membrane**: thin signal at the outermost cell boundary, rendered
  as a large-radius thin elliptical outline enclosing the nucleus.
- **Mitochondria**: punctate elongated organelles distributed in the
  cytoplasmic region, rendered as scattered small ellipses of variable
  orientation.
- **Endoplasmic reticulum**: perinuclear reticular network, rendered as
  concentric ellipses with radial spokes emanating from the nucleus.

All images were generated at 224 × 224 × 3 pixels with additive Gaussian
noise (σ = 8) to simulate photon shot noise characteristic of fluorescence
acquisition. The final dataset comprised 1,800 images (300 per class),
yielding a perfectly balanced corpus with a Gini impurity of 0.8333.

### 2.2 Dataset Partitioning

A stratified 70/15/15 train/validation/test split was applied using
StratifiedShuffleSplit, preserving the class distribution exactly in each
partition (210/45/45 images per class respectively). Normalisation
statistics (mean and standard deviation per channel) were computed
exclusively from the training split to prevent any form of data leakage
into the evaluation pipeline.

### 2.3 Augmentation Strategy

A domain-specific augmentation pipeline was constructed using the
Albumentations library. Biological cells are orientation-invariant — there
is no canonical spatial orientation for subcellular structures — justifying
aggressive geometric augmentations including random 90° rotations, horizontal
and vertical flips, and affine transformations (shift, scale, rotate).
Elastic deformation and grid distortion were applied stochastically to
simulate the morphological variability observed in real cell populations.
Intensity augmentations, including Gaussian noise, ISO noise, random
brightness-contrast perturbation, and CLAHE, simulated acquisition
variability from photon shot noise, detector read noise, and photobleaching.
Coarse dropout (random rectangular occlusion) was applied at 30% probability
to encourage robustness to partial occlusion, as is common in densely packed
cell cultures.

---

## 3. Model Architectures

### 3.1 CellNet — Custom CNN with Residual SE Blocks

CellNet is a purpose-built CNN comprising 4,376,934 trainable parameters
organised into a stem, four residual stages, and a classification head.

The stem applies a 7 × 7 convolution with stride 2 followed by max-pooling,
establishing a broad initial receptive field well-suited to capturing
whole-cell morphology. Four progressive stages expand the channel depth
from 32 to 512, with spatial resolution halved at stages 2, 3, and 4.
Stages 1–3 employ pre-activation residual blocks (He et al., 2016 v2), where
the BN → ReLU → Conv ordering improves gradient flow compared to standard
post-activation residual connections. Each residual block incorporates a
Squeeze-and-Excitation (SE) module (Hu et al., 2018) that performs
channel-wise feature recalibration: global average pooling compresses each
feature map to a scalar, a two-layer MLP with bottleneck ratio 16 learns
channel importance weights, and the resulting sigmoid-gated scale factors
are applied element-wise to the feature maps. This allows the network to
suppress uninformative channels (e.g. background) and amplify discriminative
channels (e.g. the nuclear signal in fluorescence images) adaptively.

Stage 4 employs depthwise separable convolutions (Howard et al., 2017),
reducing parameter count by factorising the standard convolution into a
depthwise spatial convolution (one filter per input channel) and a pointwise
1 × 1 convolution for channel mixing. Global average pooling replaces a
fully-connected spatial layer, providing spatial invariance and reducing
overfitting. A dropout layer (p = 0.40) precedes the final linear classifier.

Weight initialisation followed the Kaiming normal scheme for convolutions
(He et al., 2015), Xavier uniform for linear layers, and constant
initialisation for batch normalisation parameters.

### 3.2 CBAM-ResNet50 — Attention-Augmented Transfer Learning

CBAM-ResNet50 integrates a ResNet50 backbone (He et al., 2016), pretrained
on ImageNet-1k with IMAGENET1K_V2 weights, with Convolutional Block Attention
Modules (CBAM; Woo et al., 2018) inserted at the output of each residual
stage. The total parameter count is 25,321,558, of which 23,508,032
constitute the pretrained backbone.

CBAM applies two sequential attention mechanisms. Channel attention first
compresses each feature map through both global average pooling and global
max pooling, passes each descriptor through a shared two-layer MLP with
bottleneck ratio 16, sums the outputs, and applies sigmoid gating to produce
a channel-wise importance vector. This allows the model to selectively
emphasise feature channels encoding discriminative biological structures.
Spatial attention subsequently concatenates the channel-averaged and
channel-max-pooled feature maps, applies a 7 × 7 convolution, and generates
a spatial gating map indicating which image regions are most relevant for
the classification decision. The combination of channel and spatial attention
provides a comprehensive, structured mechanism for directing model focus.

The classification head comprises two linear layers with batch normalisation,
ReLU activations, and dropout (p = 0.40 and p = 0.20 respectively), providing
a deeper bottleneck for domain adaptation from ImageNet features to
fluorescence microscopy statistics.

Training followed a two-phase strategy informed by Raghu et al. (2019).
In Phase 1 (epochs 1–15), the backbone was frozen and only the CBAM modules
and classification head were trained at a learning rate of 1 × 10⁻³. This
prevented the randomly initialised head from corrupting pretrained backbone
weights via large gradient updates in early training. In Phase 2 (epoch 16
onwards), the backbone was unfrozen and the entire network was fine-tuned
end-to-end at a reduced learning rate of 1 × 10⁻⁴.

---

## 4. Training Configuration

Both models were trained using the AdamW optimiser (Loshchilov & Hutter,
2019) with decoupled weight decay (λ = 1 × 10⁻⁴). Label smoothing
cross-entropy (ε = 0.10; Szegedy et al., 2016) was used as the loss
function, providing regularisation by preventing the model from assigning
arbitrarily high confidence to training labels and improving output
calibration. A cosine annealing schedule with warm restarts (SGDR;
Loshchilov & Hutter, 2017) was applied to the learning rate, with
T₀ = 10 epochs and multiplicative period extension T_mult = 2. Periodic
restarts encourage exploration of the loss landscape and help the optimiser
escape local minima. Mixed precision training (IEEE 754 float16 with
dynamic loss scaling via GradScaler) reduced VRAM consumption and accelerated
training on the Tesla T4 GPU. Gradient clipping with maximum norm 1.0 was
applied at every step to prevent gradient explosion during fine-tuning.
Early stopping was applied with patience 12 and minimum improvement
threshold 1 × 10⁻⁴ on validation loss.

---

## 5. Results

### 5.1 Training Dynamics

CellNet converged rapidly from scratch, achieving 100% validation accuracy
by epoch 4 and stabilising thereafter. Early stopping triggered at epoch 38,
with the best checkpoint at epoch 30 (val loss 0.4236). The close alignment
of training and validation curves throughout training — with no sustained
divergence — indicates that the augmentation and regularisation strategy
effectively controlled overfitting on the 1,260-sample training set.

CBAM-ResNet50 exhibited a characteristic transfer learning signature in Phase
1: validation accuracy exceeded training accuracy during the early epochs,
as pretrained features generalised immediately before the classification head
had fully adapted. The Phase 2 transition at epoch 16 produced a brief
training loss bump as the backbone gradients were reintroduced, followed by
smooth monotonic descent. The model trained for the full 60 epochs without
early stopping, reaching best val loss 0.4214 at epoch 55.

### 5.2 Test Set Performance

Both models achieved 99.63% accuracy on the 270-sample held-out test set,
with macro ROC-AUC of 1.0000, Cohen's κ of 0.9956, and MCC of 0.9956.
Per-class analysis revealed that CellNet's single misclassification involved
an endoplasmic reticulum image predicted as plasma membrane — biologically
plausible given the structural similarity between the ring-like ER cisternae
and the plasma membrane boundary in the synthetic images. CBAM-ResNet50's
single error involved a nucleoplasm image predicted as cytosol, reflecting
the visual overlap between diffuse nuclear fill and cytoplasmic signal when
cells are closely packed.

### 5.3 Explainability Analysis

Grad-CAM and Grad-CAM++ attribution maps demonstrated clear biological
concordance for both models. Nucleoplasm attributions concentrated on nuclear
interiors; nuclear membrane attributions highlighted ring boundaries; cytosol
attributions covered the annular region between the nucleus and cell edge;
plasma membrane attributions focused on the outermost boundary; mitochondria
attributions tracked the elongated punctate distribution; and endoplasmic
reticulum attributions captured the concentric perinuclear ring pattern and
radial spokes. This concordance provides evidence that both models are
learning biologically meaningful spatial features rather than spurious
background correlations or image generation artefacts.

Quantitative spatial entropy analysis of Grad-CAM++ maps revealed that
CBAM-ResNet50 produced lower entropy (more spatially focused attribution)
than CellNet across all six classes. Mean entropy: CellNet 14.93,
CBAM-ResNet50 14.44. This difference is attributable to the explicit spatial
attention mechanism in CBAM, which directly conditions the feature maps on
spatially localised signals before the classification head is applied.
CellNet's SE blocks provide channel-level recalibration only, without
explicit spatial gating, resulting in broader, more diffuse attribution maps.

### 5.4 Statistical Comparison

McNemar's test on paired predictions yielded a test statistic of 1.0000
and p-value of 1.0 (two-sided), providing no evidence to reject the null
hypothesis of equal error rates. Only 2 discordant pairs were observed out
of 270 test samples — one in each direction — confirming statistical
equivalence. Bootstrap resampling (N = 10,000) estimated the 95% confidence
interval on the accuracy difference at [-0.0111, +0.0111], symmetrically
straddling zero. Wilson score confidence intervals for both models were
identical: [0.9793, 0.9993]. Per-class McNemar's tests were non-significant
for all six classes.

---

## 6. Discussion

The primary finding of this work is that a carefully designed 4.4M-parameter
custom CNN trained from scratch achieves statistically indistinguishable
performance from a 25.3M-parameter pretrained attention model on this
classification task. This result is interpretable within the framework of
Raghu et al. (2019), who showed that transfer learning advantages diminish
as within-domain data sufficiency increases, and that the inductive biases
of CNNs are well-matched to structured image classification tasks with
limited training data.

The key differentiator between the models is not predictive accuracy but
interpretability quality. CBAM-ResNet50's systematically lower Grad-CAM++
spatial entropy across all classes reflects a more precise and structured
attention mechanism. In real biological imaging contexts — where the
scientific value of a model depends partly on the credibility of its
explanations to domain experts — this difference may be more consequential
than raw accuracy differences. A model that correctly identifies the
endoplasmic reticulum and concentrates its attribution on the perinuclear
reticular network is more scientifically trustworthy than one achieving the
same accuracy through a broader, less interpretable feature response.

The convergence of both models to equivalent performance also validates the
quality of the synthetic dataset generation. The six morphological signatures
are distinct enough to be learned reliably by models of varying architectural
complexity, confirming that the procedural generation strategy captured the
essential discriminative structure of fluorescence microscopy data.

---

## 7. Limitations

Several limitations of the present work should be acknowledged.

**Synthetic data**: The dataset was generated procedurally rather than
acquired from real biological specimens. While the morphological patterns
were designed to reflect genuine subcellular structures, real fluorescence
microscopy data exhibits substantially greater morphological heterogeneity,
cell-to-cell variability, multi-label co-localisation, background
autofluorescence, and acquisition noise profiles that our Gaussian model
does not capture. Performance on real data is expected to be lower and would
require systematic evaluation.

**Dataset scale**: With 300 images per class, the dataset is small by deep
learning standards. The high accuracy achieved here reflects the visual
distinctiveness of the synthetic classes. On real HPA or RxRx1 data with
thousands of morphologically similar conditions, more sophisticated
regularisation, self-supervised pre-training, or semi-supervised approaches
would likely be necessary.

**Class complexity**: The six classes considered here represent clean,
single-compartment localisations. In practice, proteins frequently exhibit
multi-compartment localisation patterns (e.g. cytosol + nucleus), requiring
multi-label classification frameworks not addressed in this work.

**Evaluation depth**: With only 45 test samples per class, per-class metric
estimates carry wide confidence intervals. A larger held-out evaluation set
would provide more stable per-class ROC-AUC and F1 estimates.

---

## 8. Future Work

The following directions represent natural extensions of this pipeline:

- **Real data evaluation**: Application of both models to the Human Protein
  Atlas Single Cell Image Classification dataset or the RxRx1 dataset would
  provide external validation and test generalisation from synthetic to real
  fluorescence microscopy data.

- **Self-supervised pre-training**: Contrastive learning objectives
  (SimCLR, MoCo, DINO) applied to unlabelled microscopy images would provide
  domain-specific initialisation for CellNet, potentially closing the
  performance gap with transfer learning under data-limited conditions.

- **Multi-label extension**: Replacing the softmax classifier with a sigmoid
  multi-label head and training with binary cross-entropy would allow
  prediction of co-localisation patterns, more reflective of real biological
  protein behaviour.

- **Vision Transformer comparison**: With a larger dataset, a data-efficient
  ViT (DeiT; Touvron et al., 2021) or hybrid CNN-ViT architecture could be
  evaluated to assess whether global self-attention provides interpretability
  or accuracy advantages over the local attention of CBAM.

- **Uncertainty quantification**: Monte Carlo Dropout (Gal & Ghahramani,
  2016) or deep ensembles (Lakshminarayanan et al., 2017) would provide
  calibrated prediction uncertainty — critical for deployment in automated
  screening pipelines where model confidence must inform downstream decisions.

- **Morphological feature correlation**: Systematic correlation of Grad-CAM
  attribution maps with ground-truth morphological annotations (e.g. nucleus
  segmentation masks) would provide a quantitative measure of biological
  plausibility beyond the qualitative visual inspection conducted here.

---

## 9. Conclusions

This project demonstrated that deep learning models can achieve high accuracy
in cellular state classification from fluorescence microscopy images across
six biologically defined subcellular localisation classes. CellNet, a novel
architecture with residual SE blocks and depthwise separable convolutions,
achieved 99.63% test accuracy without pretrained weights. CBAM-ResNet50, an
attention-augmented transfer learning model, achieved identical accuracy with
the added benefit of more spatially precise and biologically interpretable
Grad-CAM attribution maps. Statistical analysis confirmed that neither model
is superior in predictive accuracy, supporting the conclusion that
architectural complexity beyond a well-regularised custom CNN does not
yield measurable performance gains on datasets of this scale and morphological
distinctiveness.

The Grad-CAM analysis revealed that both models attend to biologically
meaningful spatial regions in making their predictions, providing a measure
of scientific interpretability beyond raw performance metrics. This
interpretability — the concordance between model attention and known
subcellular anatomy — is a prerequisite for confident deployment of deep
learning tools in biological image analysis pipelines.

---

## References

Caicedo, J.C., et al. (2017). Data-analysis strategies for image-based
cell profiling. *Nature Methods*, 14(9), 849–863.

Chattopadhay, A., et al. (2018). Grad-CAM++: Generalized Gradient-based
Visual Explanations for Deep Convolutional Networks. *WACV*.

Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers
for image recognition at scale. *ICLR*.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
*ICML*.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
MIT Press.

He, K., et al. (2016). Deep Residual Learning for Image Recognition.
*CVPR*. / Identity Mappings in Deep Residual Networks. *ECCV*.

Howard, A.G., et al. (2017). MobileNets: Efficient Convolutional Neural
Networks for Mobile Vision Applications. *arXiv:1704.01212*.

Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
*CVPR*.

Kraus, O.Z., et al. (2017). Automated analysis of high-content microscopy
data with deep learning. *Molecular Systems Biology*, 13(4).

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.

Ljosa, V., & Carpenter, A.E. (2009). Introduction to the Quantitative
Analysis of Microscopy Images. *PLOS Computational Biology*.

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent
with Warm Restarts. *ICLR*.

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization.
*ICLR*.

McNemar, Q. (1947). Note on the sampling error of the difference between
correlated proportions. *Psychometrika*, 12(2), 153–157.

Muller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing
Help? *NeurIPS*.

Raghu, M., et al. (2019). Transfusion: Understanding Transfer Learning for
Medical Imaging. *NeurIPS*.

Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization. *ICCV*.

Sullivan, D.P., et al. (2018). Deep learning is combined with massive-scale
citizen science to improve large-scale image classification. *Nature
Biotechnology*, 36(9), 820–828.

Szegedy, C., et al. (2016). Rethinking the Inception Architecture.
*CVPR*.

Touvron, H., et al. (2021). Training data-efficient image transformers.
*ICML*.

Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
