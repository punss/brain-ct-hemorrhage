# Feature Importance Analysis

Source model: **Logistic Regression** (highest val AUC: 0.6976). Coefficients are from the standardised feature space — magnitudes are comparable across features. Sign indicates direction: positive = predicts presence, negative = predicts absence.

Plots:
- `group_importance.png` — mean |coefficient| and relative importance per group per label
- `top_features_per_label.png` — top 12 features by |coefficient| per label
- `lr_lda_coef_heatmap.png` — full 269-feature coefficient patterns for LR and LDA side by side

---

## Feature Group Importance (relative, row-normalised)

| Label | Histogram | Stats | Rings | Regions | Cross-ch | GLCM |
|-------|:---------:|:-----:|:-----:|:-------:|:--------:|:----:|
| epidural | 0.155 | 0.094 | 0.168 | **0.236** | 0.184 | 0.163 |
| intraparenchymal | 0.166 | 0.093 | **0.215** | 0.244 | 0.144 | 0.138 |
| intraventricular | 0.152 | 0.078 | **0.198** | **0.255** | 0.086 | 0.231 |
| subarachnoid | **0.175** | 0.107 | 0.154 | 0.228 | 0.178 | 0.159 |
| subdural | **0.185** | **0.137** | 0.165 | 0.184 | 0.132 | 0.197 |

**Regions** (location and shape of bright blobs) is the dominant group for every label. Beyond that, each label has a distinct secondary signature.

---

## Per-Label Analysis

### Epidural

Epidural hemorrhage sits between the skull and the dura, accumulating in the peripheral zone of the image.

**Top features driving presence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `sd_hist_b02` | Histogram | +0.190 |
| `bb_n_comps` | Regions | +0.176 |
| `mc_n_comps` | Regions | +0.169 |
| `sd_hist_b01` | Histogram | +0.147 |
| `bw_compactness` | Regions | +0.133 |

**Top features driving absence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `bw_ring0_max` | Rings | −0.183 |
| `diff_bb_sd_p95` | Cross-ch | −0.173 |
| `mc_glcm_homogeneity` | GLCM | −0.171 |
| `sd_p95` | Stats | −0.143 |
| `bw_ring1_max` | Rings | −0.140 |

**Interpretation:** Epidural blood is peripheral and often takes the form of multiple scattered bright fragments (`n_comps` positive). Very low intensity bins in the subdural window (`sd_hist_b01`, `sd_hist_b02`) being positive is counter-intuitive at first but makes sense: epidural hemorrhage does not fill the subdural space — the subdural window picks up mostly dark pixels, and this darkness is itself a discriminating signal. Strong central brightness (`bw_ring0_max`) and a homogeneous max-contrast signal (`mc_glcm_homogeneity`) both predict *absence* — these are signatures of deeper hemorrhage types.

Cross-channel difference (`diff_bb_sd_p95`) being negative is notable: a large brain-bone vs subdural window difference at high intensity means the subdural window is capturing bright signal that brain-bone does not, pointing to surface blood — which is subdural, not epidural.

---

### Intraparenchymal

Intraparenchymal hemorrhage is located within brain tissue, in the mid zones of the image.

**Top features driving presence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `mc_glcm_homogeneity` | GLCM | +0.239 |
| `bw_ring0_max` | Rings | +0.189 |
| `bw_ring1_max` | Rings | +0.167 |
| `bw_hist_b06` | Histogram | +0.181 |
| `mc_hist_b30` | Histogram | +0.165 |

**Top features driving absence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `mc_n_comps` | Regions | −0.185 |
| `bw_compactness` | Regions | −0.149 |
| `mc_hist_b10` | Histogram | −0.144 |
| `sd_hist_b02` | Histogram | −0.131 |
| `bb_ring0_std` | Rings | −0.108 |

**Interpretation:** `mc_glcm_homogeneity` being the single strongest feature makes anatomical sense: intraparenchymal hemorrhage is typically a compact, homogeneous bright mass sitting within brain tissue. The ring features confirm the depth — `bw_ring0_max` and `bw_ring1_max` (inner and mid-inner zones) being strongly positive places the hemorrhage centrally. Many connected components (`mc_n_comps` negative) and high compactness (`bw_compactness` negative) both predict *absence* — intraparenchymal hemorrhage is usually one coherent mass, not scattered fragments.

---

### Intraventricular

Intraventricular hemorrhage fills the brain's ventricles, which sit in the center of the image.

**Top features driving presence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `bw_ring0_max` | Rings | +0.210 |
| `bw_ring0_mean` | Rings | +0.121 |
| `mc_glcm_homogeneity` | GLCM | +0.120 |
| `bw_ring1_max` | Rings | +0.119 |
| `mc_ring0_mean` | Rings | +0.113 |

**Top features driving absence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `bw_compactness` | Regions | −0.129 |
| `sd_hist_b02` | Histogram | −0.124 |
| `bb_compactness` | Regions | −0.118 |
| `sd_compactness` | Regions | −0.113 |
| `bb_ring0_std` | Rings | −0.112 |

**Interpretation:** The most anatomically clean result in the dataset. Ring 0 (center zone) features dominate entirely — `bw_ring0_max` is the single strongest feature overall, and multiple ring 0 statistics appear in the top 5. The ventricles are centrally located and fill with bright blood homogeneously, which explains why `mc_glcm_homogeneity` is also strongly positive. Compactness being consistently negative across all channels suggests intraventricular blood is not a compact blob but diffuse fill within the ventricular space.

This label's high AUC (0.854) directly reflects how cleanly the ring decomposition captures the anatomical signal.

---

### Subarachnoid

Subarachnoid hemorrhage spreads diffusely through the CSF spaces and sulci, with no dominant localization.

**Top features driving presence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `bb_hist_b17` | Histogram | +0.109 |
| `mc_ring0_max` | Rings | +0.106 |
| `sd_n_comps` | Regions | +0.099 |
| `diff_bb_sd_p95` | Cross-ch | +0.089 |
| `bb_hist_b02` | Histogram | +0.087 |

**Top features driving absence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `mc_cy_norm` | Regions | −0.127 |
| `bw_hist_b05` | Histogram | −0.102 |
| `sd_hist_b21` | Histogram | −0.095 |
| `bb_hist_b27` | Histogram | −0.094 |
| `bw_ring0_std` | Rings | −0.089 |

**Interpretation:** Subarachnoid is the hardest label to interpret because it has no single dominant anatomical zone — coefficients are smaller and more spread across groups than any other label. Histogram features dominate (relative importance 0.175), consistent with the diffuse nature of the hemorrhage: the overall intensity distribution shifts rather than a localized bright blob appearing. Multiple components in the subdural window (`sd_n_comps` positive) captures the fragmented, filamentous appearance of blood in the sulci. The cross-channel difference being positive suggests subarachnoid blood creates a detectable signal gap between the brain-bone and subdural windows at high intensities. The low overall AUC (0.605) reflects that these subtle, diffuse features are genuinely difficult to separate from noise with Tier 1 features.

---

### Subdural

Subdural hemorrhage accumulates between the dura and the brain surface, forming a thin crescent across the outer brain.

**Top features driving presence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `sd_p95` | Stats | +0.159 |
| `bw_hist_b13` | Histogram | +0.086 |
| `bw_hist_b03` | Histogram | +0.080 |
| `mc_hist_b25` | Histogram | +0.074 |
| `mc_cy_norm` | Regions | +0.074 |

**Top features driving absence:**

| Feature | Group | Coefficient |
|---------|-------|:-----------:|
| `mc_glcm_homogeneity` | GLCM | −0.139 |
| `mc_hist_b27` | Histogram | −0.114 |
| `bw_ring0_mean` | Rings | −0.112 |
| `bb_hist_b13` | Histogram | −0.104 |
| `mc_hist_b13` | Histogram | −0.100 |

**Interpretation:** The subdural window 95th percentile (`sd_p95`) is the single strongest positive feature — the subdural window was radiologically designed to highlight the subdural space, and a high 95th percentile directly captures bright blood in that region. The Stats group has its highest relative importance here (0.137) compared to other labels, reflecting that global intensity statistics in the subdural window are meaningful. `mc_glcm_homogeneity` being strongly negative is the mirror of intraparenchymal — subdural blood is a thin irregular crescent, not a homogeneous mass. The `mc_cy_norm` being positive (bright region centroid shifted downward in the max-contrast window) is consistent with subdural accumulation at the base of the brain.

---

## Cross-label Patterns

### `mc_glcm_homogeneity` — the most polarising feature

| Label | Coefficient | Meaning |
|-------|:-----------:|---------|
| intraparenchymal | +0.239 | Compact, homogeneous mass in mid-brain |
| intraventricular | +0.120 | Homogeneous fill in ventricles |
| subdural | −0.139 | Thin irregular crescent, not homogeneous |
| epidural | −0.171 | Fragmented peripheral blood |

A single scalar from the GLCM effectively separates deep/central hemorrhages from surface ones.

### `bw_ring0_max` — central zone signal

| Label | Coefficient |
|-------|:-----------:|
| intraventricular | +0.210 |
| intraparenchymal | +0.189 |
| epidural | −0.183 |

High intensity in the center of the brain strongly predicts intraventricular and intraparenchymal, and strongly contradicts epidural.

### `bw_compactness` — blob shape

Consistently negative across intraparenchymal, intraventricular, and epidural. A compact bright region (high compactness) does not predict any of these types — they all manifest as irregular, diffuse, or multi-region patterns rather than tight blobs.

### Subarachnoid and subdural share weak, diffuse signal

Both labels have smaller maximum coefficients (~0.11–0.16) than intraparenchymal and intraventricular (~0.21–0.24). Their AUC reflects this: features engineered from intensity, texture, and radial zones struggle to capture thin surface hemorrhages that don't produce a dominant local signal. Non-linear models (Tier 2/3) are expected to close this gap.
