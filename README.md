### MRI-Based IDH Prediction in Diffuse Glioma
### Overview
This project evaluates whether IDH mutation status (wild/mutant) can be predicted using pre-operative structural MRI scan data. Utilising the UCSF-PDGM dataset, a controlled comparison between a radiomics logistic-regression baseline and a 2.5D CNN was run. The primary finding was that the radiomics logistic-regression model remained the strongest overall model on the held-out test set.
This problem is relevant because the modern diffuse-glioma classification now relies primarily on tumour molecular identity (IDH status, MGMT promoter methylation, and more). MRI data presents a key opportunity here, then, as it is available pre-operatively and usually before definitive tissue-based molecular characterisation is complete. Its role in being acting as a predictor for IDH status, which can help differentiate an astrocytoma from a glioblastoma, for example, has been well documented in recent research. At the same time, however, MRI-based prediction should be understood as a non-invasive adjunct rather than a replacement for pathology.
This project was aimed at understanding this broader radiogenomics effort, to connect tumour biology, imaging phenotype, and clinical decision-making potential, without overestimating what MRI alone can tell us.
### Dataset and Frozen Cohort
The underlying imaging resource was UCSF-PDGM, a public pre-operative diffuse-glioma MRI dataset released to support AI applications in glioma research. The original release paper describes a standardized 3T pre-operative brain tumour MRI protocol, tumour segmentation, and IDH status for all cases, with MGMT status available for a subset of higher-grade tumours (which was ignored for simplicity of v1).
For this repository, I first audited the local release rather than assuming the archive layout. The audited package contained 501 visit directories, which resolved to 495 patient-level primary pre-operative subjects after excluding 6 explicit follow-up directories from the primary cohort. The v1 study was then frozen to 495 subjects, with IDH as the only target label, complete FLAIR + T1c + T2 availability, and complete tumour segmentation across the frozen cohort. The final IDH class counts were 392 wildtype and 103 mutant.
A key design decision was to define the region-of-interest (ROI) as the whole-tumour binary union from tumor_segmentation > 0, rather than allowing the models to learn from arbitrary background anatomy. This made the analysis lesion-focused and ensured that both radiomics and CNN inputs were derived from the same frozen tumour definition.

### Methods
Cohorts were split on the level of patients, and modality selection, image preprocessing, ROI definition, and the train/test split were fixed before the main modelling stage; mainly to keep the later comparison between classical machine learning and deep learning fair.
For preprocessing, I used the canonical bias-corrected structural inputs for FLAIR, T1c, and T2. Because the frozen structural volumes already shared the same 240 × 240 × 155 grid at 1 mm isotropic resolution, no additional resampling was introduced. Each modality was normalized per image using non-zero voxels only, clipping to the [1, 99] percentile range and then z-scoring. Tumour-centred padded crops were defined from the tight whole-tumour bounding box.
The primary modelling path was a radiomics + logistic regression baseline. A compact radiomics table of 80 features was extracted from normalized FLAIR, T1c, and T2 within the binary whole-tumour ROI. These features included shape, first-order intensity statistics, and simple filter-response terms. The primary classifier was a class-weighted logistic regression trained only on the frozen radiomics table, with model selection performed entirely inside the training pool using predefined patient-level cross-validation folds. A bounded random-forest comparator was also included.
I then implemented a compact 2.5D CNN under the same frozen cohort, modality set, ROI rule, preprocessing scope, and held-out test split. CNN inputs were deterministic tumour-centred tensors built from five axial slices across the three locked modalities, yielding (15, 128, 128) inputs. The CNN was kept small rather than heavily optimized, so that it functioned as a controlled image-based comparison rather than an architecture-search exercise.
### Results
The primary radiomics logistic-regression baseline achieved the strongest held-out test performance overall. On the fixed 99-subject test set, it reached ROC-AUC 0.941, balanced accuracy 0.895, sensitivity 0.905, and specificity 0.885. Under the same experimental contract, the bounded random forest was weaker overall, and the compact 2.5D CNN reached similar ROC-AUC territory but materially lower balanced accuracy and sensitivity.
			Model
			ROC-AUC
			Balanced Accuracy
			Sensitivity
			Specificity
			Logistic regression
			0.941
			0.895
			0.905
			0.885
			Random forest
			0.914
			0.855
			0.762
			0.949
			Compact 2.5D CNN
			0.932
			0.754
			0.571
			0.936
This comparison showed that a simpler and more interpretable radiomics model captured the meaningful IDH associated signal more effectively than the bounded CNN. Repeated stratified 5-fold cross-validation on the training pool, repeated 20 times, gave a mean validation ROC-AUC of 0.961, with a mean balanced accuracy of 0.892. To quantify uncertainty on the fixed test set without reusing it for tuning, I also computed 5,000 stratified bootstrap resamples of the frozen logistic test predictions. The resulting 95% confidence intervals were 0.864 to 0.995 for ROC-AUC and 0.817 to 0.955 for balanced accuracy.
These were important because the held-out test set contained only 21 IDH-mutant subjects, so uncertainty was considerable even with strong point estimates. Nevertheless, the robustness analysis showed this was a stable and credible internal result, not simply one favourable train/test split.
### Interpretability
Predictive interpretability was addressed. For the radiomics model, I examined feature stability across repeated training-pool resamples. The most repeatedly top-ranked features included t1c_skewness, t1c_std, shape_sphericity_approx, t1c_grad_std, t1c_min, t2_iqr, t2_max, and flair_laplace_mean, with the top features showing high sign consistency across repeated fits. The dominance of T1c intensity features makes biological sense as IDH-mutant gliomas typically exhibit reduced contrast enhancement compared to IDH-wildtype tumours, potentially due to differences in angiogenesis and blood-brain barrier integrity.
For the CNN, I generated a small Grad-CAM panel spanning representative true-positive, true-negative, false-positive, and false-negative cases. These saliency maps provide a qualitative view of where the image model concentrated its attention, but they should be interpreted as model diagnostics, and not as mechanistic explanations of glioma biology. More generally, feature stability and Grad-CAM in this repository help explain how the models made predictions under the frozen internal evaluation, but they do not establish causal imaging biomarkers or reveal the molecular mechanism of IDH-mutant glioma.
### Limitations
This study is a single-cohort internal evaluation only, with no external validation cohort. The held-out test set also contains a limited number of mutant cases, which is why the confidence intervals are important. The radiomics extractor is intentionally compact rather than a full IBSI-style implementation, and the CNN was evaluated as one bounded comparison configuration rather than a broad deep-learning optimization campaign. No MGMT modelling, all-modality ablations, survival modelling, or 3D CNN comparisons are included. Most importantly, neither the feature-stability results nor the saliency maps should be used to infer mechanistic tumour biology, and none of the reported results should be taken as evidence of clinical readiness.
### Repository Structure
A simplified repository structure is:
- data/interim/ — frozen cohort, modality QC, splits
- data/processed/ — preprocessing indices, ROI masks, radiomics features, CNN inputs, bootstrap/robustness summaries
- reports/ — audit notes, cohort freeze, preprocessing design, baseline results, robustness reports, CNN comparison reports, and figure assets
- src/glioma_idh/ — ingestion, preprocessing, radiomics, baseline modelling, robustness analysis, and CNN comparison code
- configs/ — frozen cohort, preprocessing, and robustness configuration files
### Next Steps
External validation on an independent cohort, further calibration and threshold analysis, and carefully justified multimodal extensions, could take this enrich the project.

####AI Acknowledgement
This project was developed with coding assistance of OpenAI Codex. 
All experimental design decisions, biological interpretation, and written documentation are my own.