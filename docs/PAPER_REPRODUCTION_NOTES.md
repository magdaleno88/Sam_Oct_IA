# Reproduction notes: Li et al. (2019)

Reference: Feng Li et al., “Deep learning-based automated detection of retinal diseases using
optical coherence tomography images,” *Biomedical Optics Express* 10(12), 6204–6226 (2019),
DOI `10.1364/BOE.10.006204`.

## A. Explicitly described by the paper

- Four classes: CNV, DME, DRUSEN, and NORMAL.
- ImageNet-pretrained improved ResNet50 with dilated convolution.
- Four independently initialized/order-randomized models and arithmetic mean of probabilities.
- 224 x 224 inputs, horizontal reflection, random cropping, and Gaussian-pyramid downsampling.
- Participant-level separation, 10-fold cross-validation, learning rate 1e-5, batch size 200,
  and 10,000 training steps.
- Independent test participants and occlusion analysis; the supplied prompt identifies the
  reproduced occlusion window as 28 x 28.

## B. Inferred implementation decisions

- AdamW, cross-entropy, weight decay, ReduceLROnPlateau, early stopping, mixed precision,
  gradient accumulation, dropout, and exact dilation tuple are configurable decisions.
- Torchvision's `replace_stride_with_dilation` preserves official ResNet weight compatibility.
- Softmax is applied only during prediction or ensemble aggregation.
- A conservative crop scale of 0.9–1.0 replaces manually reviewed random crops.

## C. Changes required for Kermany/UCSD

- Folder labels replace hospital records. The loader accepts official train/val/test folders or
  a flat four-class folder.
- The official test split is preserved. Validation comes only from official training.
- Patient identifiers are accepted only for the recognizable `CLASS-case-image` filename form.
- Image-level splitting is disabled by default and, when explicitly enabled, is marked
  `image-level split, risk of leakage`.

## D. Methodological improvements

- Test data never selects models or hyperparameters.
- SHA-256 duplicate detection and patient-overlap checks fail before training.
- Corrupt/suspect images are reported, not deleted by arbitrary thresholds.
- Every run records config, environment, Git commit, manifest hashes, seed, checkpoints,
  predictions, figures, logs, metrics, and dataset summary.
- Ensemble inference is sequential to reduce GPU memory pressure.
- Grad-CAM is available only as a complementary method and is not presented as paper reproduction.

## E. Elements not reproducible here

- The private Shanghai hospital dataset, patient histories, manual crop review, original hardware,
  exact source code, exact optimizer/loss, and fully specified dilation topology are unavailable.
- The Kermany dataset is absent from this checkout. No clinical training or performance result has
  been produced. Automated tests use clearly synthetic images.

## F. Leakage risks

- Filename-derived patient IDs may vary between public mirrors and must be inspected in the audit.
- Different encodings of the same scan evade exact hashes; perceptual duplicates are not detected.
- Image-level bootstrap intervals overstate precision when several B-scans belong to one patient.
- Official Kermany folders do not by themselves prove patient independence.

## G. Comparing results

The paper reports results from different acquisition equipment, institutions, prevalence, and
patient selection. Metrics from this implementation must be labelled as Kermany results and must
not be presented as direct replication of the reported hospital-dataset performance. No result
may be reported until real manifests, checkpoints, and saved predictions exist.
