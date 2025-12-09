---
layout: chapter
title: "Chapter 27: Multi-Modal Learning for Clinical AI"
chapter_number: 27
part_number: 7
prev_chapter: /chapters/chapter-26-llms-in-healthcare/
next_chapter: /chapters/chapter-28-continual-learning/
---
# Chapter 27: Multi-Modal Learning for Clinical AI

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Understand the mathematical foundations of multimodal learning architectures and their application to heterogeneous healthcare data
2. Implement production-ready multimodal fusion strategies including early, late, and intermediate fusion with attention mechanisms
3. Design robust systems that handle missing modalities, a pervasive challenge in clinical settings where data availability varies systematically
4. Apply interpretability methods specific to multimodal models to understand cross-modal interactions and individual modality contributions
5. Evaluate fairness properties across modalities and demographic groups, recognizing that data availability itself may be a source of systematic bias
6. Deploy multimodal clinical AI systems that maintain performance and equity in resource-constrained environments where certain modalities may be unavailable

## Introduction: The Multimodal Nature of Clinical Data

Healthcare is fundamentally multimodal. A single patient encounter generates structured electronic health record (EHR) data including vital signs and laboratory values, unstructured clinical notes documenting the provider's assessment, medical imaging from radiological studies, time-series physiological monitoring from bedside devices, genomic sequences, and audio recordings of heart sounds or patient-reported outcomes. This rich tapestry of complementary information holds tremendous promise for comprehensive clinical decision support, yet it also poses significant technical and ethical challenges.

Traditional machine learning approaches have largely focused on single modalities, training separate models for chest radiographs, clinical notes, or laboratory values. However, clinicians naturally integrate information across modalities when making diagnostic and therapeutic decisions. A radiologist interpreting a chest X-ray considers the clinical indication documented in the ordering provider's note. An intensivist assessing hemodynamic stability combines vital sign trends with laboratory values and physical examination findings. This multimodal reasoning is not merely additive but synergistic, with information from one modality informing the interpretation of another.

The promise of multimodal learning in healthcare extends beyond improved predictive performance. By learning joint representations across data types, multimodal models can potentially identify novel disease phenotypes, detect subtle patterns invisible within single modalities, and provide more robust predictions when individual data sources are noisy or incomplete. Yet realizing this promise requires confronting several fundamental challenges. Healthcare data modalities have vastly different statistical properties: images are high-dimensional and spatially structured, text is discrete and sequential, laboratory values are low-dimensional and often missing, and time-series data exhibit complex temporal dependencies. Combining these heterogeneous representations in a principled manner requires careful architectural design and training procedures.

From an equity perspective, multimodal learning presents both opportunities and risks. On one hand, integrating diverse data sources may reduce reliance on any single modality that might be systematically biased or unavailable for certain populations. For instance, a model that can make reasonable predictions using either imaging or laboratory data provides flexibility when one modality is inaccessible. On the other hand, multimodal systems risk amplifying disparities if certain modalities are systematically unavailable in under-resourced settings. A model trained primarily on data from tertiary care centers with advanced imaging capabilities may perform poorly in community health centers where such imaging is unavailable. Moreover, the requirement for multiple modalities may create barriers to deployment in precisely the settings where clinical decision support is most needed.

This chapter develops the mathematical foundations and practical implementations of multimodal learning for clinical AI, with particular attention to the challenges of missing modalities and fairness across populations with differential data availability. We begin with core multimodal architectures and fusion strategies, then address the critical problem of missing modalities through both architectural innovations and training procedures. We develop interpretability methods specific to multimodal systems and establish evaluation frameworks that assess fairness not only across demographic groups but also across patterns of data availability. Throughout, we emphasize production-ready implementations suitable for deployment in diverse clinical environments.

## Mathematical Foundations of Multimodal Learning

Consider a clinical prediction task where we have access to $$M $$ different modalities. For a given patient $$ i $$, we observe data $$\mathbf{x}_i = \{\mathbf{x}_i^{(1)}, \mathbf{x}_i^{(2)}, \ldots, \mathbf{x}_i^{(M)}\}$$ where $$\mathbf{x}_i^{(m)}$$ represents the data from modality $$m$$. Each modality lives in its own space: $$\mathbf{x}_i^{(m)} \in \mathcal{X}^{(m)}$$. For instance, in a clinical setting we might have imaging data $$\mathbf{x}_i^{(1)} \in \mathbb{R}^{H \times W \times C}$$ (a chest radiograph), clinical notes $$\mathbf{x}_i^{(2)} \in \mathcal{V}^{L}$$ (a sequence of tokens from vocabulary $$\mathcal{V}$$ of length $$L$$), and structured EHR data $$\mathbf{x}_i^{(3)} \in \mathbb{R}^{D}$$ (laboratory values and vital signs).

The goal of multimodal learning is to learn a function $$ f: \mathcal{X}^{(1)} \times \mathcal{X}^{(2)} \times \cdots \times \mathcal{X}^{(M)} \rightarrow \mathcal{Y}$$ that maps from the joint space of all modalities to an output space $$\mathcal{Y}$$, such as disease diagnosis or mortality risk. The key challenge is that the modalities are heterogeneous: they have different dimensionalities, statistical properties, and semantic content.

### Modality-Specific Encoders

The first step in multimodal learning is encoding each modality into a common representational space. We define modality-specific encoders $$\phi^{(m)}: \mathcal{X}^{(m)} \rightarrow \mathbb{R}^{d}$$ that map raw modality data to fixed-dimensional embeddings. These encoders are typically deep neural networks whose architecture is tailored to the structure of each modality. For imaging data, convolutional neural networks extract spatial features. For text, transformer-based language models encode semantic content. For structured data, feedforward networks or graph neural networks (when relationships between features are known) produce embeddings.

Formally, for each modality $$ m$$, we compute an embedding:

$$
\mathbf{z}_i^{(m)} = \phi^{(m)}(\mathbf{x}_i^{(m)}; \theta^{(m)})
$$

where $$\theta^{(m)}$$ are the parameters of the encoder for modality $$ m $$, and $$\mathbf{z}_i^{(m)} \in \mathbb{R}^{d}$$ is the resulting embedding. The choice of embedding dimension $$d$$ involves trade-offs: larger dimensions provide more representational capacity but increase computational cost and risk of overfitting, particularly when training data is limited.

### Fusion Strategies

Once we have embeddings for each modality, we must combine them to make predictions. The multimodal fusion strategy determines how information is integrated across modalities. There are three primary approaches: early fusion, late fusion, and intermediate fusion.

**Early Fusion** concatenates raw or lightly processed features from all modalities before learning a joint representation:

$$
\mathbf{h}_i = \psi([\mathbf{x}_i^{(1)}, \mathbf{x}_i^{(2)}, \ldots, \mathbf{x}_i^{(M)}]; \theta_{\text{fusion}})
$$

where $$[\cdot]$$ denotes concatenation (or another combining operation), and $$\psi$$ is a joint model that processes all modalities simultaneously. Early fusion allows the model to learn low-level cross-modal interactions but requires all modalities to be available during training and inference. In clinical settings, this can be problematic as the concatenation of a high-resolution image with a short text note creates imbalanced inputs.

**Late Fusion** processes each modality independently through modality-specific models, then combines their predictions:

$$
\mathbf{z}_i^{(m)} = \phi^{(m)}(\mathbf{x}_i^{(m)}; \theta^{(m)})
$$

$$
\hat{y}_i^{(m)} = g^{(m)}(\mathbf{z}_i^{(m)}; \theta_g^{(m)})
$$

$$
\hat{y}_i = \text{Aggregate}(\{\hat{y}_i^{(1)}, \hat{y}_i^{(2)}, \ldots, \hat{y}_i^{(M)}\})
$$

where $$g^{(m)}$$ is a prediction head for modality $$ m$$, and the aggregation function might be averaging, voting, or a learned weighted combination. Late fusion is robust to missing modalities since each modality produces an independent prediction, but it cannot capture cross-modal interactions that may be clinically important. For instance, a late fusion model cannot learn that certain imaging findings are particularly concerning in the context of specific laboratory abnormalities.

**Intermediate Fusion** strikes a balance by learning modality-specific encoders, then fusing their embeddings to learn joint representations:

$$
\mathbf{z}_i^{(m)} = \phi^{(m)}(\mathbf{x}_i^{(m)}; \theta^{(m)})
$$

$$
\mathbf{h}_i = \psi([\mathbf{z}_i^{(1)}, \mathbf{z}_i^{(2)}, \ldots, \mathbf{z}_i^{(M)}]; \theta_{\text{fusion}})
$$

$$
\hat{y}_i = g(\mathbf{h}_i; \theta_g)
$$

This approach allows learning both modality-specific representations and cross-modal interactions while providing some flexibility when modalities are missing, though typically requiring additional strategies to handle missing data robustly.

### Attention-Based Fusion

Attention mechanisms have emerged as powerful tools for multimodal fusion, allowing the model to dynamically weight the contribution of each modality based on the input. Given embeddings $$\{\mathbf{z}_i^{(1)}, \mathbf{z}_i^{(2)}, \ldots, \mathbf{z}_i^{(M)}\}$$, we compute attention weights $$\alpha_i^{(m)}$$ that indicate the importance of modality $$m$$ for predicting the outcome of patient $$i$$:

$$
\alpha_i^{(m)} = \frac{\exp(w^{(m)\top} \mathbf{z}_i^{(m)})}{\sum_{m'=1}^{M} \exp(w^{(m')\top} \mathbf{z}_i^{(m')})}
$$

The fused representation is then a weighted combination:

$$
\mathbf{h}_i = \sum_{m=1}^{M} \alpha_i^{(m)} \mathbf{z}_i^{(m)}
$$

More sophisticated attention mechanisms compute attention weights based on interactions between modalities. Cross-modal attention allows one modality to attend to another:

$$
\mathbf{q}_i = W_q \mathbf{z}_i^{(1)}, \quad \mathbf{k}_i^{(m)} = W_k \mathbf{z}_i^{(m)}, \quad \mathbf{v}_i^{(m)} = W_v \mathbf{z}_i^{(m)}
$$

$$
\alpha_i^{(m)} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_i^{(m)} / \sqrt{d})}{\sum_{m'=2}^{M} \exp(\mathbf{q}_i^\top \mathbf{k}_i^{(m')} / \sqrt{d})}
$$

$$
\mathbf{h}_i = \mathbf{z}_i^{(1)} + \sum_{m=2}^{M} \alpha_i^{(m)} \mathbf{v}_i^{(m)}
$$

This formulation, inspired by transformer architectures, allows imaging data (modality 1) to query information from clinical notes (modalities 2 through $$M $$), creating rich cross-modal representations. The attention weights $$\alpha_i^{(m)}$$ also provide interpretability, indicating which modalities were most informative for each prediction.

### Co-Attention and Transformer-Based Fusion

Co-attention mechanisms extend cross-modal attention by allowing bidirectional information flow between modalities. For two modalities with embeddings $$\mathbf{z}_i^{(1)} \in \mathbb{R}^{d_1}$$ and $$\mathbf{z}_i^{(2)} \in \mathbb{R}^{d_2}$$, co-attention computes:

$$
\mathbf{C}_i = \tanh(\mathbf{z}_i^{(1)} W_c (\mathbf{z}_i^{(2)})^\top)
$$

where $$\mathbf{C}_i \in \mathbb{R}^{d_1 \times d_2}$$ captures pairwise affinities between elements of the two modalities. Attention weights are derived from $$\mathbf{C}_i$$ to produce modality-specific attended representations that encode information from both modalities.

Transformer architectures have been adapted for multimodal learning by treating embeddings from different modalities as different types of tokens in a sequence. Given modality embeddings, we add modality-specific positional encodings:

$$
\tilde{\mathbf{z}}_i^{(m)} = \mathbf{z}_i^{(m)} + \mathbf{e}^{(m)}
$$

where $$\mathbf{e}^{(m)}$$ is a learned modality embedding. These tokens are then processed by transformer layers:

$$
\mathbf{Z}_i^{(l+1)} = \text{TransformerLayer}(\mathbf{Z}_i^{(l)})
$$

where $$\mathbf{Z}_i^{(l)} = [\tilde{\mathbf{z}}_i^{(1)}, \tilde{\mathbf{z}}_i^{(2)}, \ldots, \tilde{\mathbf{z}}_i^{(M)}]$$ at layer $$l$$. The transformer's self-attention mechanism allows each modality to attend to all others, learning rich cross-modal interactions. The final prediction is typically derived from a special classification token or by pooling across all modality tokens.

### Joint Embedding Spaces and Contrastive Learning

An alternative approach to multimodal learning focuses on learning a joint embedding space where semantically similar inputs from different modalities are mapped to nearby points. This is particularly valuable when we have paired multimodal data but limited labeled outcomes. For instance, we might have many chest X-rays with associated radiology reports but few with disease labels.

Contrastive learning trains encoders to maximize agreement between different views of the same entity while minimizing agreement between different entities. For modalities $$ m $$ and $$ m'$$ with paired data $$(\mathbf{x}_i^{(m)}, \mathbf{x}_i^{(m')})$$, we compute embeddings and maximize their similarity while minimizing similarity to negative pairs:

$$
\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i^{(m)}, \mathbf{z}_i^{(m')}) / \tau)}{\sum_{j \neq i} \exp(\text{sim}(\mathbf{z}_i^{(m)}, \mathbf{z}_j^{(m')}) / \tau)}
$$

where $$\text{sim}(\cdot, \cdot)$$ is a similarity function (typically cosine similarity), and $$\tau $$ is a temperature parameter. This loss, known as InfoNCE, encourages the encoders to produce aligned representations across modalities. In clinical settings, this allows learning from image-text pairs (such as radiographs and reports) without explicit labels, then fine-tuning on smaller labeled datasets for specific tasks.

The CLIP (Contrastive Language-Image Pre-training) framework has demonstrated remarkable success in learning joint vision-language representations. Adaptations for clinical data, such as BiomedCLIP and RadFM, learn alignments between medical images and clinical text, enabling zero-shot or few-shot prediction on new tasks by computing similarity between image embeddings and text descriptions of diseases.

## Architectures for Clinical Multimodal Learning

### Vision-Language Models for Medical Imaging

Medical imaging reports naturally pair images with text, making vision-language models particularly applicable. The standard architecture consists of an image encoder (typically a convolutional neural network or vision transformer), a text encoder (typically a transformer-based language model), and a mechanism to align their representations.

For a chest radiograph $$\mathbf{x}_{\text{img}}$$ and associated report $$\mathbf{x}_{\text{text}}$$, we compute:

$$
\mathbf{z}_{\text{img}} = \phi_{\text{img}}(\mathbf{x}_{\text{img}}), \quad \mathbf{z}_{\text{text}} = \phi_{\text{text}}(\mathbf{x}_{\text{text}})
$$

These embeddings are projected to a joint space:

$$
\mathbf{u}_{\text{img}} = W_{\text{img}} \mathbf{z}_{\text{img}}, \quad \mathbf{u}_{\text{text}} = W_{\text{text}} \mathbf{z}_{\text{text}}
$$

The contrastive loss aligns paired image-text embeddings. Once trained, the model can be used for various downstream tasks. For classification, we compute similarity between the image embedding and text embeddings of disease descriptions. For report generation, the image embedding serves as context for an autoregressive text decoder.

Clinical vision-language models face unique challenges compared to natural images. Medical images often contain subtle pathological findings requiring high-resolution processing. Reports contain domain-specific terminology and follow semi-structured templates. Moreover, reports may be incomplete or focus on normal findings, providing weak supervision.

### Multimodal Fusion for Structured EHR Data

Electronic health records integrate diverse structured data: demographics, vital signs, laboratory values, medications, and procedures. While these are all "structured," they have different statistical properties. Laboratory values are continuous but often missing, medications are categorical with complex interactions, and procedures occur at irregular time points.

A typical multimodal EHR architecture processes each data type separately before fusion. Continuous variables pass through feedforward networks with normalization:

$$
\mathbf{z}_{\text{lab}} = \phi_{\text{lab}}(\mathbf{x}_{\text{lab}})
$$

Categorical variables are embedded:

$$
\mathbf{z}_{\text{med}} = \sum_{j} \mathbf{E}_{\text{med}}[x_{\text{med},j}]
$$

where $$\mathbf{E}_{\text{med}}$$ is a learned embedding matrix. Time-series data (vital signs, treatments over time) are processed by recurrent networks or transformers:

$$
\mathbf{z}_{\text{ts}} = \text{RNN}(\{\mathbf{x}_{\text{ts}}^{(t_1)}, \mathbf{x}_{\text{ts}}^{(t_2)}, \ldots\})
$$

These embeddings are fused using attention or concatenation:

$$
\mathbf{h} = \text{FusionModule}([\mathbf{z}_{\text{lab}}, \mathbf{z}_{\text{med}}, \mathbf{z}_{\text{ts}}])
$$

The fusion module might be a feedforward network, attention mechanism, or graph neural network that models relationships between data types.

### Integrating Imaging, Text, and Structured Data

The most challenging multimodal scenarios combine fundamentally different data types: images, unstructured text, and structured data. Consider predicting acute kidney injury in hospitalized patients using chest radiographs, clinical notes, and laboratory values. Each modality provides complementary information: imaging reveals pulmonary edema (a complication of fluid overload), notes document urine output and medication exposure, and laboratory values quantify renal function.

A comprehensive architecture processes each modality with appropriate encoders:

$$
\mathbf{z}_{\text{img}} = \phi_{\text{img}}(\mathbf{x}_{\text{img}}; \theta_{\text{img}})
$$

$$
\mathbf{z}_{\text{text}} = \phi_{\text{text}}(\mathbf{x}_{\text{text}}; \theta_{\text{text}})
$$

$$
\mathbf{z}_{\text{lab}} = \phi_{\text{lab}}(\mathbf{x}_{\text{lab}}; \theta_{\text{lab}})
$$

A cross-modal transformer fuses these embeddings:

$$
\mathbf{Z} = [\mathbf{z}_{\text{img}} + \mathbf{e}_{\text{img}}, \mathbf{z}_{\text{text}} + \mathbf{e}_{\text{text}}, \mathbf{z}_{\text{lab}} + \mathbf{e}_{\text{lab}}]
$$

$$
\mathbf{H} = \text{Transformer}(\mathbf{Z}; \theta_{\text{trans}})
$$

$$
\hat{y} = \text{softmax}(W_{\text{out}} \text{Pool}(\mathbf{H}) + b_{\text{out}})
$$

The modality embeddings $$\mathbf{e}_{\text{img}}, \mathbf{e}_{\text{text}}, \mathbf{e}_{\text{lab}}$$ inform the transformer which modality each token represents. The pooling operation aggregates information across modalities, typically using the first token's output (if a CLS token is prepended) or average pooling.

## Handling Missing Modalities in Clinical Settings

Missing modalities are pervasive in clinical settings. Not all patients receive imaging studies; laboratory tests are ordered based on clinical indication; advanced diagnostics may be unavailable in resource-constrained settings. A multimodal model that requires all modalities during inference is impractical for clinical deployment. Moreover, missingness is often not random: certain modalities are systematically absent in particular populations or care settings, creating potential for bias.

### Architectural Strategies for Missing Modalities

**Modality Dropout During Training:** A simple but effective approach introduces modality dropout during training. With probability $$ p_{\text{drop}}^{(m)}$$, we set the embedding for modality $$ m $$ to a learned mask token or zero vector:

$\tilde{\mathbf{z}}_i^{(m)} = \begin{cases}
\mathbf{z}_i^{(m)} & \text{with probability } 1 - p_{\text{drop}}^{(m)} \\
\mathbf{z}_{\text{mask}}^{(m)} & \text{with probability } p_{\text{drop}}^{(m)}
\end{cases}$

This forces the model to learn representations that remain informative even when certain modalities are absent. The dropout probabilities should reflect the expected missingness patterns at deployment: if imaging is available for only fifty percent of patients, set $$ p_{\text{drop}}^{(\text{img})} = 0.5$$.

**Modality-Specific Adapters:** Another approach trains a universal multimodal model with modality-specific adapter networks that activate only when the modality is present:

$$
\mathbf{h}_i = \sum_{m=1}^{M} \mathbb{1}[m \in \mathcal{M}_i] \cdot \text{Adapter}^{(m)}(\mathbf{z}_i^{(m)})
$$

where $$\mathcal{M}_i$$ is the set of available modalities for patient $$i$$, and $$\mathbb{1}[\cdot]$$ is an indicator function. Each adapter is a small feedforward network that transforms modality-specific embeddings to a common space. This architecture naturally handles any combination of available modalities without requiring retraining.

**Mixture of Modality Experts:** We can train separate expert models for each modality subset, then route each patient to the appropriate expert based on available modalities. For $$ M $$ modalities, this requires training $$ 2^M - 1$$ models (one for each non-empty subset). While computationally expensive during training, this approach allows each expert to optimize for its specific input pattern. A gating network can dynamically weight expert predictions when multiple experts are applicable:

$$
\hat{y}_i = \sum_{s \in \mathcal{S}_i} \alpha_{i,s} \cdot \hat{y}_{i,s}
$$

where $$\mathcal{S}_i$$ is the set of experts compatible with patient $$i$$'s available modalities, and $$\alpha_{i,s}$$ are learned gating weights.

### Generative Approaches for Missing Modalities

Rather than handling missing modalities during prediction, we can attempt to impute the missing data. Generative models learn the conditional distribution of one modality given others, allowing synthesis of missing modalities.

For instance, given a patient's clinical notes and laboratory values, we might generate a plausible chest radiograph:

$$
\hat{\mathbf{x}}_{\text{img}} \sim p(\mathbf{x}_{\text{img}} \mid \mathbf{x}_{\text{text}}, \mathbf{x}_{\text{lab}}; \theta_{\text{gen}})
$$

This generated image can then be used by the multimodal model. Generative adversarial networks (GANs), variational autoencoders (VAEs), and diffusion models have all been explored for cross-modal generation in medical imaging.

However, this approach requires caution in clinical settings. Generated images may contain hallucinated findings that mislead downstream predictions. If the generative model introduces systematic artifacts, these can propagate to the multimodal classifier. Moreover, generating missing modalities may obscure the fact that data was unavailable, which itself may be clinically informative.

A more conservative approach uses generative models to estimate uncertainty about the missing modality. Rather than generating a single imputation, we sample multiple plausible imputations and propagate this uncertainty through the prediction:

$$
\hat{y}_i = \mathbb{E}_{\mathbf{x}_{\text{img}} \sim p(\mathbf{x}_{\text{img}} \mid \mathbf{x}_{\text{text}}, \mathbf{x}_{\text{lab}})}[f(\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{text}}, \mathbf{x}_{\text{lab}})]
$$

This Monte Carlo estimate provides both a point prediction and uncertainty bounds reflecting the missing information.

### Fairness Implications of Missing Modalities

Missingness patterns often correlate with demographic characteristics and socioeconomic status. Patients in rural areas may have limited access to advanced imaging. Safety-net hospitals may lack the resources for comprehensive laboratory panels. Certain populations may face barriers to completing diagnostic workups due to cost, transportation, or language barriers.

If a multimodal model's performance degrades significantly when certain modalities are missing, and those modalities are systematically absent in underserved populations, the model will exhibit disparate performance. We must therefore evaluate models across all relevant missingness patterns and demographic groups.

Define a missingness pattern $$\mathcal{M} \subseteq \{1, 2, \ldots, M\}$$ as a subset of available modalities. For each pattern $$\mathcal{M}$$ and demographic group $$g$$, we compute performance metrics:

$$
\text{Performance}(\mathcal{M}, g) = \text{Metric}(\hat{\mathbf{y}}[\mathcal{M}_i = \mathcal{M}, G_i = g], \mathbf{y}[\mathcal{M}_i = \mathcal{M}, G_i = g])
$$

where $$[\cdot]$$ denotes subsetting. A model is fair across missingness patterns if performance is similar across groups for each missingness pattern:

$$
\lvert \text{Performance}(\mathcal{M}, g) - \text{Performance}(\mathcal{M}, g') \rvert < \epsilon \quad \forall \mathcal{M}, g, g'
$$

This criterion is stronger than typical fairness metrics because it requires equitable performance not only across groups but also across patterns of data availability.

## Interpretability for Multimodal Models

Understanding how multimodal models make predictions is essential for clinical adoption. Clinicians need to know which modalities and features drove a prediction, both to verify clinical plausibility and to identify potential errors or biases.

### Attention Visualization

When using attention-based fusion, the attention weights $$\alpha^{(m)}$$ indicate the relative importance of each modality for a given prediction. Visualizing these weights provides a high-level summary of the model's reasoning:

$$
\text{Importance}^{(m)} = \frac{\alpha^{(m)}}{\sum_{m'} \alpha^{(m')}}
$$

For a patient with chest radiograph, clinical notes, and laboratory values, we might find that the model assigns seventy percent weight to imaging, twenty percent to notes, and ten percent to labs, suggesting that imaging findings were the primary driver of the prediction.

However, attention weights alone do not provide complete interpretability. High attention on a modality does not necessarily mean that modality caused the prediction; attention reflects what the model attends to, not necessarily what it uses. Moreover, attention operates on learned embeddings whose meaning may not align with clinical concepts.

### Cross-Modal Gradients

Gradient-based methods can identify which features within each modality contribute most to predictions. For a multimodal model $$f(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(M)})$$, the gradient of the output with respect to inputs in modality $$ m$$ indicates feature importance:

$$
\mathbf{g}^{(m)} = \frac{\partial f}{\partial \mathbf{x}^{(m)}}
$$

For imaging modalities, these gradients can be visualized as saliency maps highlighting image regions that influenced the prediction. For text, gradients indicate which words or phrases were most important. For structured data, gradients quantify the sensitivity of predictions to each feature.

Cross-modal gradients can also reveal interactions between modalities. The mixed partial derivative:

$$
\frac{\partial^2 f}{\partial \mathbf{x}^{(m)} \partial \mathbf{x}^{(m')}}
$$

captures how changing a feature in one modality affects the importance of features in another modality, revealing cross-modal synergies.

### Integrated Gradients for Multimodal Attribution

Integrated gradients provide a principled attribution method that satisfies desirable properties including sensitivity (if changing a feature changes the prediction, the feature receives non-zero attribution) and implementation invariance (attribution depends only on the function implemented by the model, not its architecture).

For each modality, we compute integrated gradients by accumulating gradients along a path from a baseline input to the actual input:

$$
\text{IG}^{(m)}(\mathbf{x}^{(m)}) = (\mathbf{x}^{(m)} - \mathbf{x}_{\text{baseline}}^{(m)}) \times \int_{\alpha=0}^{1} \frac{\partial f}{\partial \mathbf{x}^{(m)}}(\mathbf{x}_{\text{baseline}}^{(m)} + \alpha (\mathbf{x}^{(m)} - \mathbf{x}_{\text{baseline}}^{(m)})) d\alpha
$$

The choice of baseline is important and modality-specific. For images, we might use a black image or Gaussian noise. For text, we might replace words with mask tokens. For structured data, we might use population means or zeros.

In practice, the integral is approximated by a finite sum over steps:

$$
\text{IG}^{(m)}(\mathbf{x}^{(m)}) \approx (\mathbf{x}^{(m)} - \mathbf{x}_{\text{baseline}}^{(m)}) \times \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f}{\partial \mathbf{x}^{(m)}}(\mathbf{x}_{\text{baseline}}^{(m)} + \frac{k}{n} (\mathbf{x}^{(m)} - \mathbf{x}_{\text{baseline}}^{(m)}))
$$

This provides feature-level attributions within each modality while naturally handling multimodal inputs.

### Modality Ablation Studies

A complementary approach to interpretability systematically removes or masks each modality and measures the change in predictions. For patient $$i $$ with prediction $$\hat{y}_i$$ using all modalities, we compute predictions using all subsets:

$$
\hat{y}_i^{(-m)} = f(\mathbf{x}_i^{(1)}, \ldots, \mathbf{x}_i^{(m-1)}, \mathbf{0}, \mathbf{x}_i^{(m+1)}, \ldots, \mathbf{x}_i^{(M)})
$$

The importance of modality $$m$$ is quantified by the change in prediction:

$$
\Delta_i^{(m)} = \lvert \hat{y}_i - \hat{y}_i^{(-m)} \rvert
$$

This can be extended to measure the importance of modality combinations by ablating multiple modalities simultaneously. Shapley values provide a principled framework for computing feature importance that satisfies desirable axioms (efficiency, symmetry, dummy, additivity). For modality $$m$$:

$$
\phi^{(m)} = \sum_{\mathcal{S} \subseteq \mathcal{M} \setminus \{m\}} \frac{\lvert \mathcal{S} \rvert! (M - \lvert \mathcal{S} \rvert - 1)!}{M!} [f(\mathcal{S} \cup \{m\}) - f(\mathcal{S})]
$$

where $$\mathcal{M}$$ is the set of all modalities, and $$ f(\mathcal{S})$$ is the model's prediction using only modalities in set $$\mathcal{S}$$. Computing exact Shapley values requires evaluating $$ 2^M$$ subsets, but efficient approximations exist.

### Concept-Based Explanations

High-level clinical concepts may not align with individual features within modalities. Concept-based interpretability methods learn or define clinically meaningful concepts, then assess how much each concept contributed to a prediction.

For a predefined set of clinical concepts $$\{c_1, c_2, \ldots, c_K\}$$ (such as "pulmonary edema," "elevated creatinine," "history of heart failure"), we train concept detectors that identify the presence of each concept from multimodal inputs:

$$
p(c_k \mid \mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(M)}) = g_k(\phi^{(1)}(\mathbf{x}^{(1)}), \ldots, \phi^{(M)}(\mathbf{x}^{(M)}))
$$

We then train a final model that predicts the outcome from detected concepts:

$$
\hat{y} = h(p(c_1), \ldots, p(c_K))
$$

This two-stage approach provides interpretable explanations in terms of clinical concepts: "The model predicts high risk of acute kidney injury because it detected pulmonary edema in the chest X-ray (concept score 0.9) and elevated creatinine in the labs (concept score 0.85)."

Concepts can be defined by domain experts or discovered automatically through concept bottleneck models that jointly learn concepts and predictions while encouraging concepts to be human-interpretable.

## Implementation: Production-Ready Multimodal Clinical AI

We now develop comprehensive implementations of multimodal clinical AI systems, emphasizing robustness, fairness evaluation, and handling of missing modalities.

### Multimodal Data Loading and Preprocessing

```python
"""
Multimodal Clinical Data Pipeline
Production-ready data loading for heterogeneous healthcare data with comprehensive
preprocessing, missingness handling, and equity-aware validation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
import logging
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Enumeration of supported modality types."""
    IMAGE = "image"
    TEXT = "text"
    STRUCTURED = "structured"
    TIMESERIES = "timeseries"

@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    modality_type: ModalityType
    is_required: bool
    preprocessing: Optional[Dict[str, Any]] = None
    embedding_dim: int = 256

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")

@dataclass
class MultimodalSample:
    """Container for a single multimodal patient sample."""
    patient_id: str
    modalities: Dict[str, torch.Tensor]
    available_modalities: List[str]
    label: Optional[torch.Tensor]
    demographic_group: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate sample consistency."""
        for modality_name in self.available_modalities:
            if modality_name not in self.modalities:
                raise ValueError(
                    f"Modality {modality_name} listed as available but not present in data"
                )

class ClinicalImagePreprocessor:
    """
    Preprocessor for medical imaging data with domain-specific augmentation
    and normalization suitable for deployment.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        augment: bool = True
    ) -> None:
        """
        Initialize image preprocessor.

        Args:
            img_size: Target image size (height, width)
            normalize_mean: Mean for normalization per channel
            normalize_std: Std for normalization per channel
            augment: Whether to apply augmentation (training only)
        """
        self.img_size = img_size
        self.augment = augment

        # Default ImageNet normalization if not specified
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

        # Build transformation pipeline
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]

        # Add augmentation for training
        if augment:
            # Conservative augmentation appropriate for medical images
            # Avoid transformations that might alter clinical findings
            transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(2, transforms.RandomRotation(degrees=5))
            transform_list.insert(
                3,
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0
                )
            )

        transform_list.append(
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        )

        self.transform = transforms.Compose(transform_list)

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply preprocessing to an image.

        Args:
            img: Input image as numpy array or tensor

        Returns:
            Preprocessed image tensor
        """
        try:
            if isinstance(img, np.ndarray):
                # Convert numpy array to PIL for torchvision transforms
                from PIL import Image
                if img.dtype != np.uint8:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                img = Image.fromarray(img)

            return self.transform(img)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

class ClinicalTextPreprocessor:
    """
    Preprocessor for clinical text data with handling of medical terminology
    and structured note formats.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 512,
        truncation_strategy: str = "longest_first"
    ) -> None:
        """
        Initialize text preprocessor.

        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
            truncation_strategy: How to truncate long sequences
        """
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            raise

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize and preprocess clinical text.

        Args:
            text: Input clinical text

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)}")

        if not text.strip():
            warnings.warn("Empty text provided, using placeholder")
            text = "[EMPTY]"

        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Remove batch dimension since we process one sample at a time
            return {k: v.squeeze(0) for k, v in encoded.items()}
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise

class StructuredDataPreprocessor:
    """
    Preprocessor for structured EHR data with imputation, normalization,
    and encoding of categorical variables.
    """

    def __init__(
        self,
        feature_names: List[str],
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        imputation_strategy: str = "median",
        normalization: str = "standard"
    ) -> None:
        """
        Initialize structured data preprocessor.

        Args:
            feature_names: List of all feature names in order
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            imputation_strategy: Strategy for missing value imputation
            normalization: Normalization strategy for numerical features
        """
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.imputation_strategy = imputation_strategy
        self.normalization = normalization

        # Initialize statistics (fit during training)
        self.feature_means: Optional[Dict[str, float]] = None
        self.feature_stds: Optional[Dict[str, float]] = None
        self.feature_medians: Optional[Dict[str, float]] = None
        self.category_mappings: Optional[Dict[str, Dict[str, int]]] = None

        self._validate_features()

    def _validate_features(self) -> None:
        """Validate feature configuration."""
        all_specified = set(self.categorical_features + self.numerical_features)
        all_features = set(self.feature_names)

        if all_specified != all_features:
            missing = all_features - all_specified
            extra = all_specified - all_features
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing specification: {missing}")
            if extra:
                msg_parts.append(f"Extra features specified: {extra}")
            raise ValueError("; ".join(msg_parts))

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit preprocessor statistics from training data.

        Args:
            data: Training dataframe with all features
        """
        if not set(self.feature_names).issubset(data.columns):
            missing = set(self.feature_names) - set(data.columns)
            raise ValueError(f"Missing features in data: {missing}")

        # Compute statistics for numerical features
        self.feature_means = {}
        self.feature_stds = {}
        self.feature_medians = {}

        for feat in self.numerical_features:
            self.feature_means[feat] = data[feat].mean()
            self.feature_stds[feat] = data[feat].std()
            self.feature_medians[feat] = data[feat].median()

        # Build category mappings for categorical features
        self.category_mappings = {}
        for feat in self.categorical_features:
            unique_vals = data[feat].dropna().unique()
            # Reserve 0 for unknown/missing
            self.category_mappings[feat] = {
                val: idx + 1 for idx, val in enumerate(sorted(unique_vals))
            }

        logger.info("Fitted structured data preprocessor")

    def __call__(self, data: Union[pd.Series, Dict[str, Any]]) -> torch.Tensor:
        """
        Preprocess structured data sample.

        Args:
            data: Single sample as Series or dict

        Returns:
            Preprocessed feature tensor
        """
        if self.feature_means is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        if isinstance(data, dict):
            data = pd.Series(data)

        processed_features = []

        # Process numerical features
        for feat in self.numerical_features:
            val = data.get(feat, np.nan)

            # Impute missing values
            if pd.isna(val):
                if self.imputation_strategy == "mean":
                    val = self.feature_means[feat]
                elif self.imputation_strategy == "median":
                    val = self.feature_medians[feat]
                elif self.imputation_strategy == "zero":
                    val = 0.0
                else:
                    raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")

            # Normalize
            if self.normalization == "standard":
                val = (val - self.feature_means[feat]) / (self.feature_stds[feat] + 1e-8)
            elif self.normalization == "minmax":
                # This would require storing min/max during fit
                raise NotImplementedError("MinMax normalization not yet implemented")

            processed_features.append(val)

        # Process categorical features
        for feat in self.categorical_features:
            val = data.get(feat, None)

            # Map to integer, use 0 for unknown
            if val is None or pd.isna(val):
                idx = 0
            else:
                idx = self.category_mappings[feat].get(val, 0)

            processed_features.append(float(idx))

        return torch.tensor(processed_features, dtype=torch.float32)

class MultimodalClinicalDataset(Dataset):
    """
    Dataset for multimodal clinical data with support for missing modalities,
    demographic annotations, and equity-aware sampling.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        modality_configs: Dict[str, ModalityConfig],
        preprocessors: Dict[str, Any],
        label_column: Optional[str] = None,
        demographic_column: Optional[str] = None,
        stratify_by_missingness: bool = True
    ) -> None:
        """
        Initialize multimodal dataset.

        Args:
            data_path: Path to data directory or manifest file
            modality_configs: Configuration for each modality
            preprocessors: Preprocessors for each modality
            label_column: Column name for labels
            demographic_column: Column name for demographic group
            stratify_by_missingness: Track missingness patterns for stratified evaluation
        """
        self.data_path = Path(data_path)
        self.modality_configs = modality_configs
        self.preprocessors = preprocessors
        self.label_column = label_column
        self.demographic_column = demographic_column
        self.stratify_by_missingness = stratify_by_missingness

        # Load manifest/index
        self.samples = self._load_manifest()
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

        # Track missingness patterns if requested
        if self.stratify_by_missingness:
            self._compute_missingness_statistics()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load dataset manifest with sample metadata."""
        # Implementation depends on data organization
        # Here we assume a CSV manifest with columns for each modality path
        manifest_path = self.data_path / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        required_cols = ["patient_id"]
        if self.label_column:
            required_cols.append(self.label_column)

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in manifest: {missing_cols}")

        return df.to_dict('records')

    def _compute_missingness_statistics(self) -> None:
        """Compute statistics about missingness patterns."""
        missingness_patterns = {}

        for sample in self.samples:
            available = []
            for modality_name in self.modality_configs.keys():
                if self._is_modality_available(sample, modality_name):
                    available.append(modality_name)

            pattern = tuple(sorted(available))
            missingness_patterns[pattern] = missingness_patterns.get(pattern, 0) + 1

        logger.info("Missingness pattern distribution:")
        for pattern, count in sorted(missingness_patterns.items(), key=lambda x: -x[1]):
            logger.info(f"  {pattern}: {count} samples ({100*count/len(self.samples):.1f}%)")

        self.missingness_patterns = missingness_patterns

    def _is_modality_available(self, sample: Dict[str, Any], modality_name: str) -> bool:
        """Check if a modality is available for a sample."""
        # Check if the path column exists and is not null
        path_col = f"{modality_name}_path"
        if path_col not in sample:
            return False
        path = sample[path_col]
        return path is not None and pd.notna(path) and Path(path).exists()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        """Load and preprocess a multimodal sample."""
        sample_info = self.samples[idx]
        patient_id = sample_info['patient_id']

        modalities = {}
        available_modalities = []

        # Process each configured modality
        for modality_name, config in self.modality_configs.items():
            if self._is_modality_available(sample_info, modality_name):
                try:
                    # Load raw data
                    path_col = f"{modality_name}_path"
                    data_path = self.data_path / sample_info[path_col]

                    if config.modality_type == ModalityType.IMAGE:
                        from PIL import Image
                        img = Image.open(data_path).convert('RGB')
                        modalities[modality_name] = self.preprocessors[modality_name](img)

                    elif config.modality_type == ModalityType.TEXT:
                        with open(data_path, 'r') as f:
                            text = f.read()
                        modalities[modality_name] = self.preprocessors[modality_name](text)

                    elif config.modality_type == ModalityType.STRUCTURED:
                        # Assume structured data is in the manifest row itself
                        modalities[modality_name] = self.preprocessors[modality_name](sample_info)

                    else:
                        raise NotImplementedError(
                            f"Modality type {config.modality_type} not implemented"
                        )

                    available_modalities.append(modality_name)

                except Exception as e:
                    logger.warning(
                        f"Failed to load {modality_name} for patient {patient_id}: {e}"
                    )
                    if config.is_required:
                        raise

            elif config.is_required:
                raise ValueError(
                    f"Required modality {modality_name} not available for patient {patient_id}"
                )

        # Load label if specified
        label = None
        if self.label_column and self.label_column in sample_info:
            label = torch.tensor(sample_info[self.label_column], dtype=torch.float32)

        # Load demographic group if specified
        demographic_group = None
        if self.demographic_column and self.demographic_column in sample_info:
            demographic_group = sample_info[self.demographic_column]

        return MultimodalSample(
            patient_id=patient_id,
            modalities=modalities,
            available_modalities=available_modalities,
            label=label,
            demographic_group=demographic_group,
            metadata=sample_info
        )

def collate_multimodal_batch(
    batch: List[MultimodalSample]
) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for multimodal batches that handles
    variable modality availability.

    Args:
        batch: List of MultimodalSample objects

    Returns:
        Dictionary with batched tensors and metadata
    """
    # Get all modality names from first sample
    all_modalities = set()
    for sample in batch:
        all_modalities.update(sample.available_modalities)

    batched = {
        'patient_ids': [s.patient_id for s in batch],
        'labels': torch.stack([s.label for s in batch]) if batch[0].label is not None else None,
        'demographic_groups': [s.demographic_group for s in batch],
        'available_modalities': [s.available_modalities for s in batch]
    }

    # Batch each modality separately
    for modality_name in all_modalities:
        modality_data = []
        modality_mask = []

        for sample in batch:
            if modality_name in sample.available_modalities:
                data = sample.modalities[modality_name]
                # Handle dict output from text preprocessor
                if isinstance(data, dict):
                    if not modality_data:
                        # Initialize dict structure
                        modality_data = {k: [] for k in data.keys()}
                    for k, v in data.items():
                        modality_data[k].append(v)
                else:
                    modality_data.append(data)
                modality_mask.append(1)
            else:
                # Create placeholder of appropriate shape
                if modality_data:
                    if isinstance(modality_data, dict):
                        placeholder = {k: torch.zeros_like(v[0]) for k, v in modality_data.items()}
                        for k, v in placeholder.items():
                            modality_data[k].append(v)
                    else:
                        placeholder = torch.zeros_like(modality_data[0])
                        modality_data.append(placeholder)
                modality_mask.append(0)

        # Stack into batch
        if isinstance(modality_data, dict):
            batched[modality_name] = {k: torch.stack(v) for k, v in modality_data.items()}
        else:
            batched[modality_name] = torch.stack(modality_data)

        batched[f'{modality_name}_mask'] = torch.tensor(modality_mask, dtype=torch.bool)

    return batched
```

### Multimodal Architecture Implementation

```python
"""
Multimodal Clinical AI Model
Production-ready implementation with attention-based fusion, missing modality handling,
and comprehensive fairness evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class ModalityEncoder(nn.Module):
    """Base class for modality-specific encoders."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1):
        """
        Initialize encoder.

        Args:
            input_dim: Input dimension for this modality
            embedding_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode modality data to embedding space."""
        raise NotImplementedError

class ImageEncoder(ModalityEncoder):
    """
    Encoder for medical imaging data using convolutional neural networks
    or vision transformers.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        architecture: str = "resnet50",
        dropout: float = 0.1
    ):
        """
        Initialize image encoder.

        Args:
            embedding_dim: Output embedding dimension
            pretrained: Whether to use pretrained weights
            architecture: CNN architecture to use
            dropout: Dropout probability
        """
        super().__init__(input_dim=2048, embedding_dim=embedding_dim, dropout=dropout)

        if architecture == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        logger.info(f"Initialized {architecture} image encoder with embedding_dim={embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to embedding.

        Args:
            x: Image tensor [batch_size, channels, height, width]

        Returns:
            Image embedding [batch_size, embedding_dim]
        """
        features = self.backbone(x)
        features = features.flatten(1)
        embedding = self.projection(features)
        return embedding

class TextEncoder(ModalityEncoder):
    """
    Encoder for clinical text using transformer-based language models.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze_base: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize text encoder.

        Args:
            embedding_dim: Output embedding dimension
            model_name: HuggingFace model name
            freeze_base: Whether to freeze base model weights
            dropout: Dropout probability
        """
        from transformers import AutoModel

        base_model = AutoModel.from_pretrained(model_name)
        base_dim = base_model.config.hidden_size

        super().__init__(input_dim=base_dim, embedding_dim=embedding_dim, dropout=dropout)

        self.base_model = base_model

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info(f"Froze base model weights for {model_name}")

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(base_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        logger.info(f"Initialized {model_name} text encoder with embedding_dim={embedding_dim}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embedding.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Text embedding [batch_size, embedding_dim]
        """
        # Get contextualized representations
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Project to embedding space
        embedding = self.projection(cls_embedding)
        return embedding

class StructuredEncoder(ModalityEncoder):
    """
    Encoder for structured EHR data.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize structured data encoder.

        Args:
            input_dim: Number of input features
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__(input_dim=input_dim, embedding_dim=embedding_dim, dropout=dropout)

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.mlp = nn.Sequential(*layers)

        logger.info(
            f"Initialized structured encoder: {input_dim} -> {hidden_dims} -> {embedding_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode structured data to embedding.

        Args:
            x: Structured features [batch_size, input_dim]

        Returns:
            Embedding [batch_size, embedding_dim]
        """
        return self.mlp(x)

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism allowing one modality to attend to another.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.

        Args:
            query: Query embeddings [batch_size, embedding_dim]
            key: Key embeddings [batch_size, num_keys, embedding_dim]
            value: Value embeddings [batch_size, num_keys, embedding_dim]
            mask: Attention mask [batch_size, num_keys]

        Returns:
            Attended embeddings and attention weights
        """
        batch_size = query.size(0)

        # Project and reshape
        q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_keys]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.embedding_dim)
        attn_output = attn_output.squeeze(1)

        output = self.out_proj(attn_output)

        return output, attn_weights.squeeze(1).squeeze(1)

class MultimodalFusionModule(nn.Module):
    """
    Fusion module combining embeddings from multiple modalities with
    attention-based weighting and cross-modal interactions.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_modalities: int,
        fusion_strategy: str = "attention",
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize fusion module.

        Args:
            embedding_dim: Dimension of modality embeddings
            num_modalities: Number of modalities
            fusion_strategy: Fusion strategy ("concat", "attention", "cross_attention")
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * num_modalities, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            )

        elif fusion_strategy == "attention":
            self.attention_weights = nn.Linear(embedding_dim, 1)
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        elif fusion_strategy == "cross_attention":
            self.cross_attention = nn.ModuleList([
                CrossModalAttention(embedding_dim, num_heads, dropout)
                for _ in range(num_modalities)
            ])
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        logger.info(f"Initialized {fusion_strategy} fusion module")

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse modality embeddings.

        Args:
            embeddings: Dictionary of modality embeddings
            masks: Dictionary of modality availability masks

        Returns:
            Fused embedding and attention weights (if applicable)
        """
        modality_names = list(embeddings.keys())
        batch_size = embeddings[modality_names[0]].size(0)

        if self.fusion_strategy == "concat":
            # Concatenate all embeddings
            concat_embeddings = torch.cat([embeddings[m] for m in modality_names], dim=1)
            fused = self.fusion(concat_embeddings)
            attention_weights = {m: None for m in modality_names}

        elif self.fusion_strategy == "attention":
            # Compute attention weights for each modality
            embeddings_stack = torch.stack([embeddings[m] for m in modality_names], dim=1)
            masks_stack = torch.stack([masks[m] for m in modality_names], dim=1)

            # Compute unnormalized attention scores
            attn_logits = self.attention_weights(embeddings_stack).squeeze(-1)
            attn_logits = attn_logits.masked_fill(~masks_stack, float('-inf'))

            # Normalize to get attention weights
            attn_weights = F.softmax(attn_logits, dim=1)

            # Weighted sum of embeddings
            weighted_sum = (embeddings_stack * attn_weights.unsqueeze(-1)).sum(dim=1)
            fused = self.fusion(weighted_sum)

            # Store attention weights for each modality
            attention_weights = {
                m: attn_weights[:, i] for i, m in enumerate(modality_names)
            }

        elif self.fusion_strategy == "cross_attention":
            # Each modality attends to all others
            attended_embeddings = []
            attention_weights = {}

            for i, modality in enumerate(modality_names):
                query = embeddings[modality]

                # Stack other modalities as key/value
                other_modalities = [m for m in modality_names if m != modality]
                keys = torch.stack([embeddings[m] for m in other_modalities], dim=1)
                values = keys

                key_masks = torch.stack([masks[m] for m in other_modalities], dim=1)

                # Apply cross-attention
                attended, attn_w = self.cross_attention[i](query, keys, values, key_masks)
                attended_embeddings.append(attended)

                attention_weights[modality] = dict(zip(other_modalities, attn_w.unbind(1)))

            # Combine attended embeddings
            combined = torch.stack(attended_embeddings, dim=1).mean(dim=1)
            fused = self.fusion(combined)

        return fused, attention_weights

class MultimodalClinicalModel(nn.Module):
    """
    Complete multimodal clinical AI model with encoders, fusion, and prediction head.
    Handles missing modalities and provides interpretability.
    """

    def __init__(
        self,
        modality_configs: Dict[str, Dict[str, Any]],
        embedding_dim: int = 512,
        fusion_strategy: str = "attention",
        num_classes: int = 2,
        dropout: float = 0.1,
        modality_dropout_prob: float = 0.2
    ):
        """
        Initialize multimodal model.

        Args:
            modality_configs: Configuration for each modality encoder
            embedding_dim: Common embedding dimension
            fusion_strategy: Strategy for fusing modalities
            num_classes: Number of output classes
            dropout: Dropout probability
            modality_dropout_prob: Probability of dropping modality during training
        """
        super().__init__()

        self.modality_names = list(modality_configs.keys())
        self.embedding_dim = embedding_dim
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.modality_dropout_prob = modality_dropout_prob

        # Initialize encoders
        self.encoders = nn.ModuleDict()
        for modality_name, config in modality_configs.items():
            encoder_type = config['type']

            if encoder_type == 'image':
                self.encoders[modality_name] = ImageEncoder(
                    embedding_dim=embedding_dim,
                    pretrained=config.get('pretrained', True),
                    architecture=config.get('architecture', 'resnet50'),
                    dropout=dropout
                )
            elif encoder_type == 'text':
                self.encoders[modality_name] = TextEncoder(
                    embedding_dim=embedding_dim,
                    model_name=config.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT'),
                    freeze_base=config.get('freeze_base', False),
                    dropout=dropout
                )
            elif encoder_type == 'structured':
                self.encoders[modality_name] = StructuredEncoder(
                    input_dim=config['input_dim'],
                    embedding_dim=embedding_dim,
                    hidden_dims=config.get('hidden_dims'),
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Initialize fusion module
        self.fusion = MultimodalFusionModule(
            embedding_dim=embedding_dim,
            num_modalities=len(self.modality_names),
            fusion_strategy=fusion_strategy,
            dropout=dropout
        )

        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )

        logger.info(
            f"Initialized multimodal model with {len(self.modality_names)} modalities, "
            f"{fusion_strategy} fusion, {num_classes} classes"
        )

    def forward(
        self,
        inputs: Dict[str, Any],
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal model.

        Args:
            inputs: Dictionary of modality inputs and masks
            return_embeddings: Whether to return intermediate embeddings
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with logits and optional embeddings/attention
        """
        batch_size = inputs[f'{self.modality_names[0]}_mask'].size(0)

        # Encode each modality
        embeddings = {}
        masks = {}

        for modality_name in self.modality_names:
            mask = inputs[f'{modality_name}_mask']

            # Apply modality dropout during training
            if self.training and torch.rand(1).item() < self.modality_dropout_prob:
                # Randomly drop this modality
                embeddings[modality_name] = torch.zeros(
                    batch_size, self.embedding_dim, device=mask.device
                )
                masks[modality_name] = torch.zeros_like(mask, dtype=torch.bool)
            else:
                # Encode modality
                modality_input = inputs[modality_name]

                if isinstance(modality_input, dict):
                    # Text modality with multiple inputs
                    embedding = self.encoders[modality_name](
                        input_ids=modality_input['input_ids'],
                        attention_mask=modality_input['attention_mask']
                    )
                else:
                    embedding = self.encoders[modality_name](modality_input)

                # Mask unavailable modalities
                embedding = embedding * mask.unsqueeze(1).float()

                embeddings[modality_name] = embedding
                masks[modality_name] = mask

        # Fuse modalities
        fused_embedding, attention_weights = self.fusion(embeddings, masks)

        # Classification
        logits = self.classifier(fused_embedding)

        # Prepare output
        outputs = {'logits': logits}

        if return_embeddings:
            outputs['embeddings'] = embeddings
            outputs['fused_embedding'] = fused_embedding

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    def get_modality_importance(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance of each modality for predictions via ablation.

        Args:
            inputs: Dictionary of modality inputs

        Returns:
            Dictionary mapping modality names to importance scores
        """
        self.eval()

        with torch.no_grad():
            # Get full model prediction
            full_output = self.forward(inputs)
            full_logits = full_output['logits']

            importances = {}

            # Ablate each modality
            for modality_name in self.modality_names:
                # Create modified inputs with this modality masked
                ablated_inputs = {k: v for k, v in inputs.items()}
                ablated_inputs[f'{modality_name}_mask'] = torch.zeros_like(
                    inputs[f'{modality_name}_mask']
                )

                # Get prediction without this modality
                ablated_output = self.forward(ablated_inputs)
                ablated_logits = ablated_output['logits']

                # Importance is the change in prediction
                importance = (full_logits - ablated_logits).abs().mean(dim=1)
                importances[modality_name] = importance

        return importances
```

### Fairness Evaluation Framework

```python
"""
Fairness evaluation for multimodal clinical AI models.
Comprehensive metrics across demographic groups and missingness patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.calibration import calibration_curve
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MultimodalFairnessEvaluator:
    """
    Evaluator for fairness metrics across demographic groups and
    modality availability patterns.
    """

    def __init__(
        self,
        demographic_groups: List[str],
        modality_patterns: Optional[List[Tuple[str, ...]]] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize fairness evaluator.

        Args:
            demographic_groups: List of demographic group identifiers
            modality_patterns: List of modality availability patterns to evaluate
            metrics: List of metrics to compute
        """
        self.demographic_groups = demographic_groups
        self.modality_patterns = modality_patterns or []

        if metrics is None:
            self.metrics = ['auroc', 'auprc', 'accuracy', 'calibration']
        else:
            self.metrics = metrics

        # Storage for predictions and labels
        self.predictions = defaultdict(list)
        self.labels = defaultdict(list)
        self.probabilities = defaultdict(list)

        logger.info(
            f"Initialized fairness evaluator for {len(demographic_groups)} groups, "
            f"{len(self.modality_patterns)} modality patterns"
        )

    def add_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
        demographic_groups: List[str],
        available_modalities: List[List[str]]
    ) -> None:
        """
        Add a batch of predictions for evaluation.

        Args:
            predictions: Predicted class labels [batch_size]
            labels: True labels [batch_size]
            probabilities: Predicted probabilities [batch_size, num_classes]
            demographic_groups: Demographic group for each sample
            available_modalities: Available modalities for each sample
        """
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        for i in range(len(predictions)):
            group = demographic_groups[i]
            modalities = tuple(sorted(available_modalities[i]))

            # Store overall
            key = ('overall', 'all_modalities')
            self.predictions[key].append(predictions[i])
            self.labels[key].append(labels[i])
            self.probabilities[key].append(probabilities[i])

            # Store by demographic group
            key = (group, 'all_modalities')
            self.predictions[key].append(predictions[i])
            self.labels[key].append(labels[i])
            self.probabilities[key].append(probabilities[i])

            # Store by modality pattern
            key = ('overall', modalities)
            self.predictions[key].append(predictions[i])
            self.labels[key].append(labels[i])
            self.probabilities[key].append(probabilities[i])

            # Store by demographic group and modality pattern
            key = (group, modalities)
            self.predictions[key].append(predictions[i])
            self.labels[key].append(labels[i])
            self.probabilities[key].append(probabilities[i])

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all fairness metrics.

        Returns:
            Nested dictionary of metrics by group and modality pattern
        """
        results = {}

        for key, preds in self.predictions.items():
            if len(preds) < 10:  # Skip groups with too few samples
                continue

            group, modality_pattern = key

            preds = np.array(preds)
            true_labels = np.array(self.labels[key])
            probs = np.array(self.probabilities[key])

            # Get positive class probabilities
            if probs.ndim == 2:
                pos_probs = probs[:, 1]
            else:
                pos_probs = probs

            metrics_dict = {}

            # AUROC
            if 'auroc' in self.metrics and len(np.unique(true_labels)) > 1:
                try:
                    auroc = roc_auc_score(true_labels, pos_probs)
                    metrics_dict['auroc'] = float(auroc)
                except Exception as e:
                    logger.warning(f"Failed to compute AUROC for {key}: {e}")

            # AUPRC
            if 'auprc' in self.metrics and len(np.unique(true_labels)) > 1:
                try:
                    auprc = average_precision_score(true_labels, pos_probs)
                    metrics_dict['auprc'] = float(auprc)
                except Exception as e:
                    logger.warning(f"Failed to compute AUPRC for {key}: {e}")

            # Accuracy
            if 'accuracy' in self.metrics:
                acc = accuracy_score(true_labels, preds)
                metrics_dict['accuracy'] = float(acc)

            # Calibration error (ECE)
            if 'calibration' in self.metrics and len(np.unique(true_labels)) > 1:
                try:
                    ece = self._compute_expected_calibration_error(true_labels, pos_probs)
                    metrics_dict['ece'] = float(ece)
                except Exception as e:
                    logger.warning(f"Failed to compute ECE for {key}: {e}")

            # Store results
            results[key] = metrics_dict

        return results

    def _compute_expected_calibration_error(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute expected calibration error.

        Args:
            labels: True binary labels
            probabilities: Predicted probabilities for positive class
            n_bins: Number of bins for calibration

        Returns:
            Expected calibration error
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracy = labels[mask].mean()
                bin_confidence = probabilities[mask].mean()
                bin_weight = mask.sum() / len(labels)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def compute_disparity_metrics(
        self,
        results: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]],
        reference_group: str = 'overall'
    ) -> Dict[str, float]:
        """
        Compute disparity metrics comparing groups to reference.

        Args:
            results: Results dictionary from compute_metrics()
            reference_group: Reference group for comparison

        Returns:
            Dictionary of disparity metrics
        """
        disparities = {}

        # Get reference metrics for each modality pattern
        reference_metrics = {}
        for key, metrics in results.items():
            group, modality_pattern = key
            if group == reference_group:
                reference_metrics[modality_pattern] = metrics

        # Compute disparities
        for key, metrics in results.items():
            group, modality_pattern = key

            if group == reference_group or modality_pattern not in reference_metrics:
                continue

            ref_metrics = reference_metrics[modality_pattern]

            for metric_name in metrics.keys():
                if metric_name not in ref_metrics:
                    continue

                # Absolute difference
                disparity_key = f"{group}_{modality_pattern}_{metric_name}_diff"
                disparities[disparity_key] = metrics[metric_name] - ref_metrics[metric_name]

                # Ratio
                if ref_metrics[metric_name] != 0:
                    disparity_key = f"{group}_{modality_pattern}_{metric_name}_ratio"
                    disparities[disparity_key] = metrics[metric_name] / ref_metrics[metric_name]

        return disparities

    def generate_report(self) -> str:
        """
        Generate a comprehensive fairness report.

        Returns:
            Formatted report string
        """
        results = self.compute_metrics()
        disparities = self.compute_disparity_metrics(results)

        report = ["=" * 80]
        report.append("MULTIMODAL FAIRNESS EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall performance
        overall_key = ('overall', 'all_modalities')
        if overall_key in results:
            report.append("Overall Performance (All Modalities):")
            for metric, value in results[overall_key].items():
                report.append(f"  {metric}: {value:.4f}")
            report.append("")

        # Performance by demographic group
        report.append("Performance by Demographic Group:")
        for group in self.demographic_groups:
            group_key = (group, 'all_modalities')
            if group_key in results:
                report.append(f"\n  {group}:")
                for metric, value in results[group_key].items():
                    report.append(f"    {metric}: {value:.4f}")
        report.append("")

        # Performance by modality pattern
        if self.modality_patterns:
            report.append("Performance by Modality Pattern:")
            for pattern in self.modality_patterns:
                pattern_key = ('overall', pattern)
                if pattern_key in results:
                    report.append(f"\n  {pattern}:")
                    for metric, value in results[pattern_key].items():
                        report.append(f"    {metric}: {value:.4f}")
            report.append("")

        # Disparity summary
        if disparities:
            report.append("Key Disparities:")
            # Show largest disparities
            sorted_disparities = sorted(
                disparities.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            for key, value in sorted_disparities:
                report.append(f"  {key}: {value:+.4f}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

def stratified_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    demographic_groups: List[str]
) -> Tuple[Dict, str]:
    """
    Perform stratified evaluation of multimodal model.

    Args:
        model: Multimodal model to evaluate
        dataloader: DataLoader with test data
        device: Device for computation
        demographic_groups: List of demographic groups

    Returns:
        Results dictionary and formatted report
    """
    model.eval()

    evaluator = MultimodalFairnessEvaluator(
        demographic_groups=demographic_groups
    )

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ['patient_ids', 'demographic_groups', 'available_modalities']
            }

            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(inputs)
            logits = outputs['logits']

            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Add to evaluator
            evaluator.add_batch(
                predictions=predictions,
                labels=labels,
                probabilities=probabilities,
                demographic_groups=batch['demographic_groups'],
                available_modalities=batch['available_modalities']
            )

    # Compute metrics and generate report
    results = evaluator.compute_metrics()
    report = evaluator.generate_report()

    return results, report
```

## Clinical Deployment Considerations

Deploying multimodal clinical AI systems requires careful attention to practical constraints and equity implications.

### Modality Availability Across Care Settings

Healthcare facilities vary enormously in available diagnostic modalities. A tertiary academic medical center may have advanced imaging, comprehensive laboratory capabilities, and genomic sequencing. A rural community health center may have basic labs and limited imaging. Mobile health clinics serving homeless populations may rely primarily on point-of-care tests and clinical examination.

If a multimodal model requires chest radiographs, comprehensive metabolic panels, and clinical notes to function, it cannot be deployed in settings where these are unavailable. This creates a fundamental equity problem: the most sophisticated AI tools become available only in the most resourced settings, exacerbating rather than reducing disparities.

Design principles for equitable multimodal systems include graceful degradation, where performance declines gradually rather than catastrophically as modalities become unavailable, uncertainty quantification that increases as fewer modalities are available but never prevents deployment, and transparent reporting of which modality combinations were used for each prediction. A model should clearly indicate "This prediction used imaging and labs" versus "This prediction used only labs due to unavailable imaging," allowing clinicians to appropriately calibrate their trust.

### Computational Considerations

Multimodal models are computationally expensive. Processing high-resolution medical images through convolutional neural networks requires substantial GPU resources. Large language models for clinical text consume significant memory. In resource-constrained settings, these computational demands may be prohibitive.

Strategies for efficient deployment include model compression through quantization and pruning, knowledge distillation where a smaller student model is trained to mimic a larger teacher model, selective modality processing where computationally expensive modalities are only processed when clinically indicated, and edge deployment where models run locally rather than requiring cloud connectivity. The last point is particularly important for settings with unreliable internet connectivity.

### Regulatory Pathways

Multimodal clinical AI systems face complex regulatory challenges. In the United States, the FDA's framework for Software as a Medical Device (SaMD) requires demonstrating safety and effectiveness for intended use. For multimodal systems, this raises questions about whether each modality combination requires separate validation. If a model is validated using imaging plus labs, can it be deployed using only labs without additional validation?

Current regulatory frameworks are evolving to address these questions. The FDA's proposed approach for adaptive AI systems provides some guidance, but multimodal systems introduce unique complexities. International regulatory harmonization through frameworks like the International Medical Device Regulators Forum (IMDRF) may help, but significant uncertainty remains.

### Ethical Frameworks for Multimodal AI

Beyond technical performance and regulatory compliance, multimodal clinical AI raises ethical questions. When a model makes different predictions based on which modalities are available, how should clinicians and patients interpret this uncertainty? If certain populations systematically lack access to particular modalities, does deploying a multimodal system that performs better when all modalities are available constitute discrimination?

The principle of justice in medical ethics requires equitable access to beneficial interventions. If multimodal AI genuinely improves care, equity demands ensuring access across populations. This may require deliberately designing systems that work well with minimal modality requirements, even if maximal performance requires more data. It may also require policy interventions to expand access to diagnostic modalities rather than accepting current inequities as immutable.

Transparency obligations require clearly communicating to patients and clinicians which data informed each prediction. Patients have a right to know whether their care decision was based on comprehensive multimodal analysis or limited data. This transparency enables informed consent and appropriate calibration of trust.

## Future Directions and Open Challenges

Multimodal learning for clinical AI remains an active research area with many open questions. Several directions warrant particular attention given equity considerations.

### Foundation Models for Medical Multimodal Learning

Large-scale foundation models pretrained on diverse multimodal medical data may enable few-shot or zero-shot adaptation to new clinical tasks. Models like GPT-4 with vision and Med-PaLM demonstrate this potential. However, ensuring these foundation models are trained on diverse, representative data is critical. If pretraining data comes primarily from academic medical centers in high-income countries, the resulting models may not generalize to underserved populations.

### Multimodal Bias Detection and Mitigation

Current fairness metrics typically evaluate overall model performance across demographic groups. For multimodal systems, we need methods to identify which modalities contribute to bias. If a model exhibits disparate performance across groups, is this driven by biased imaging, biased text, or biased integration? Causal inference methods may help disentangle these sources of bias, enabling targeted mitigation.

### Learning with Systematically Missing Data

Standard missing data methods assume data is missing at random or missing at random conditional on observed variables (MAR). In clinical settings, missingness is often not random: certain tests are not ordered for certain populations due to structural barriers, implicit bias, or resource constraints. Learning methods that account for such systematic missingness are needed.

### Multimodal Explanation Methods

While attention mechanisms provide some interpretability, understanding multimodal predictions remains challenging. How do we explain to a clinician that a model predicted high risk because of the combination of a specific imaging finding and a particular phrase in the clinical note? Concept-based explanations may help, but developing methods that align with clinical reasoning remains an open challenge.

### Evaluation Beyond Discrimination Metrics

Standard fairness metrics focus on discrimination: ensuring similar performance across groups. However, multimodal systems raise additional equity concerns about representation (are all groups adequately represented in training data), autonomy (do patients understand and consent to multimodal AI), and social determinants (does the model account for social factors that influence both data availability and outcomes). Comprehensive evaluation frameworks that address these dimensions are needed.

## Chapter Summary

This chapter developed the mathematical foundations and practical implementations of multimodal learning for clinical AI, with particular emphasis on the challenges of missing modalities and fairness across populations with differential data availability. We covered core fusion strategies from early concatenation through sophisticated attention mechanisms, implemented production-ready multimodal architectures with comprehensive handling of missing data, and established evaluation frameworks that assess fairness not only across demographic groups but also across patterns of data availability. The equity focus throughout recognized that data availability itself is a source of systematic bias, requiring models that degrade gracefully when certain modalities are unavailable rather than failing entirely. Clinical deployment requires recognizing that the most sophisticated multimodal systems risk being accessible only in the most resourced settings, potentially exacerbating rather than reducing health disparities. Future work must ensure that multimodal AI genuinely expands access to high-quality care rather than creating new barriers based on differential availability of advanced diagnostics.

## References

1. Acosta, J. N., Falcone, G. J., Rajpurkar, P., & Topol, E. J. (2022). Multimodal biomedical AI. *Nature Medicine*, 28(9), 1773-1784.

2. Huang, S. C., Pareek, A., Seyyedi, S., Banerjee, I., & Lungren, M. P. (2020). Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines. *NPJ Digital Medicine*, 3(1), 136.

3. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning* (pp. 8748-8763). PMLR.

4. Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., & Langlotz, C. P. (2022). Contrastive learning of medical visual representations from paired images and text. In *Machine Learning for Healthcare Conference* (pp. 2-25). PMLR.

5. Bodenreider, O. (2004). The unified medical language system (UMLS): integrating biomedical terminology. *Nucleic Acids Research*, 32(suppl_1), D267-D270.

6. Banerjee, I., Bhimireddy, A. R., Burns, J. L., Celi, L. A., Chen, L. C., Correa, R., ... & Lungren, M. P. (2021). Reading race: AI recognises patient's racial identity in medical images. *Lancet Digital Health*, 3(12), e765-e766.

7. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.

8. Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423-443.

9. Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., & Ng, A. Y. (2011). Multimodal deep learning. In *International Conference on Machine Learning*.

10. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *International Conference on Machine Learning* (pp. 1321-1330). PMLR.

11. Ma, F., Gao, J., Suo, Q., You, Q., Zhou, J., & Zhang, A. (2018). Risk prediction on electronic health records with prior medical knowledge. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 1910-1919).

12. Ramachandram, D., & Taylor, G. W. (2017). Deep multimodal learning: A survey on recent advances and trends. *IEEE Signal Processing Magazine*, 34(6), 96-108.

13. Shickel, B., Tighe, P. J., Bihorac, A., & Rashidi, P. (2018). Deep EHR: A survey of recent advances in deep learning techniques for electronic health record (EHR) analysis. *IEEE Journal of Biomedical and Health Informatics*, 22(5), 1589-1604.

14. Rajpurkar, P., Chen, E., Banerjee, O., & Topol, E. J. (2022). AI in health and medicine. *Nature Medicine*, 28(1), 31-38.

15. Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L. W. H., Feng, M., Ghassemi, M., ... & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3(1), 1-9.

16. Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., ... & Dean, J. (2019). A guide to deep learning in healthcare. *Nature Medicine*, 25(1), 24-29.

17. Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M. (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144.

18. Ghassemi, M., Oakden-Rayner, L., & Beam, A. L. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*, 3(11), e745-e750.

19. Vig, J. (2019). A multiscale visualization of attention in the transformer model. *arXiv preprint arXiv:1906.05714*.

20. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In *International Conference on Machine Learning* (pp. 3319-3328). PMLR.

21. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *International Conference on Machine Learning* (pp. 2668-2677). PMLR.

22. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference* (pp. 214-226).

23. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In *Advances in Neural Information Processing Systems* (pp. 3315-3323).

24. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. In *Advances in Neural Information Processing Systems* (pp. 5680-5689).

25. Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. In *Conference on Fairness, Accountability and Transparency* (pp. 77-91). PMLR.

26. Seyyed-Kalantari, L., Zhang, H., McDermott, M. B., Chen, I. Y., & Ghassemi, M. (2021). Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. *Nature Medicine*, 27(12), 2176-2182.

27. Zhang, H., Lu, A. X., Abdalla, M., McDermott, M., & Ghassemi, M. (2020). Hurtful words: quantifying biases in clinical contextual word embeddings. In *Proceedings of the ACM Conference on Health, Inference, and Learning* (pp. 110-120).

28. Popejoy, A. B., & Fullerton, S. M. (2016). Genomics is failing on diversity. *Nature News*, 538(7624), 161.

29. Vyas, D. A., Eisenstein, L. G., & Jones, D. S. (2020). Hidden in plain sight—reconsidering the use of race correction in clinical algorithms. *New England Journal of Medicine*, 383(9), 874-882.

30. Char, D. S., Shah, N. H., & Magnus, D. (2018). Implementing machine learning in health care—addressing ethical challenges. *The New England Journal of Medicine*, 378(11), 981.

31. McCradden, M. D., Joshi, S., Anderson, J. A., Mazwi, M., Goldenberg, A., & Zlotnik Shaul, R. (2020). Patient safety and quality improvement: Ethical principles for a regulatory approach to bias in healthcare machine learning. *Journal of the American Medical Informatics Association*, 27(12), 2024-2027.

32. Gichoya, J. W., Banerjee, I., Bhimireddy, A. R., Burns, J. L., Celi, L. A., Chen, L. C., ... & Lungren, M. P. (2022). AI recognition of patient race in medical imaging: a modelling study. *The Lancet Digital Health*, 4(6), e406-e414.

33. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

34. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT* (pp. 4171-4186).

35. Huang, K., Altosaar, J., & Ranganath, R. (2019). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. *arXiv preprint arXiv:1904.05342*.

36. Alsentzer, E., Murphy, J. R., Boag, W., Weng, W. H., Jin, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. In *Clinical Natural Language Processing Workshop*.

37. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770-778).

38. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*.

39. Chen, R. J., Lu, M. Y., Wang, J., Williamson, D. F., Rodig, S. J., Lindeman, N. I., & Mahmood, F. (2021). Pathomic fusion: an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis. *IEEE Transactions on Medical Imaging*, 41(4), 757-770.

40. Lu, M. Y., Williamson, D. F., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature Biomedical Engineering*, 5(6), 555-570.

41. Kather, J. N., Pearson, A. T., Halama, N., Jäger, D., Krause, J., Loosen, S. H., ... & Yoshikawa, T. (2019). Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer. *Nature Medicine*, 25(7), 1054-1056.

42. Zhu, Y., Chen, Y., Lu, Z., Pan, S., Wang, G., Kwok, Y., & Xie, R. (2021). Multimodal transformer for multimodal machine translation. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*.

43. Tsai, Y. H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L. P., & Salakhutdinov, R. (2019). Multimodal transformer for unaligned multimodal language sequences. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 6558-6569).

44. Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In *Advances in Neural Information Processing Systems* (pp. 13-23).

45. Li, L. H., Yatskar, M., Yin, D., Hsieh, C. J., & Chang, K. W. (2019). VisualBERT: A simple and performant baseline for vision and language. *arXiv preprint arXiv:1908.03557*.

46. Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

47. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *International Conference on Machine Learning* (pp. 1597-1607). PMLR.

48. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In *IEEE Conference on Computer Vision and Pattern Recognition* (pp. 9729-9738).

49. Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., & Langlotz, C. P. (2022). Contrastive learning of medical visual representations from paired images and text. In *Machine Learning for Healthcare Conference*.

50. Müller, H., Michoux, N., Bandon, D., & Geissbuhler, A. (2004). A review of content-based image retrieval systems in medical applications—clinical benefits and future directions. *International Journal of Medical Informatics*, 73(1), 1-23.

51. Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021). WILDS: A benchmark of in-the-wild distribution shifts. In *International Conference on Machine Learning* (pp. 5637-5664). PMLR.

52. Winkler, J. K., Fink, C., Toberer, F., Enk, A., Deinlein, T., Hofmann-Wellenhof, R., ... & Haenssle, H. A. (2019). Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. *JAMA Dermatology*, 155(10), 1135-1141.

53. Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine*, 15(11), e1002683.

54. Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.

55. Little, R. J., & Rubin, D. B. (2019). *Statistical analysis with missing data* (Vol. 793). John Wiley & Sons.

56. Tran, K. A., Kondrashova, O., Bradley, A., Williams, E. D., Pearson, J. V., & Waddell, N. (2021). Deep learning in cancer diagnosis, prognosis and treatment selection. *Genome Medicine*, 13(1), 152.

57. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35.

58. Chouldechova, A., & Roth, A. (2018). The frontiers of fairness in machine learning. *arXiv preprint arXiv:1810.08810*.

59. Verma, S., & Rubin, J. (2018). Fairness definitions explained. In *IEEE/ACM International Workshop on Software Fairness* (pp. 1-7).

60. Berk, R., Heidari, H., Jabbari, S., Kearns, M., & Roth, A. (2021). Fairness in criminal justice risk assessments: The state of the art. *Sociological Methods & Research*, 50(1), 3-44.

61. Goodman, B., & Flaxman, S. (2017). European Union regulations on algorithmic decision-making and a "right to explanation". *AI Magazine*, 38(3), 50-57.

62. Wachter, S., Mittelstadt, B., & Floridi, L. (2017). Why a right to explanation of automated decision-making does not exist in the general data protection regulation. *International Data Privacy Law*, 7(2), 76-99.

63. Mittelstadt, B. D., Allo, P., Taddeo, M., Wachter, S., & Floridi, L. (2016). The ethics of algorithms: Mapping the debate. *Big Data & Society*, 3(2), 2053951716679679.

64. Zou, J., & Schiebinger, L. (2018). AI can be sexist and racist—it's time to make it fair. *Nature News*, 559(7714), 324.

65. Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183-186.

66. Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In *Advances in Neural Information Processing Systems* (pp. 4349-4357).

67. Lipton, Z. C. (2018). The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. *Queue*, 16(3), 31-57.

68. Murdoch, W. J., Singh, C., Kumbier, K., Abbasi-Asl, R., & Yu, B. (2019). Definitions, methods, and applications in interpretable machine learning. *Proceedings of the National Academy of Sciences*, 116(44), 22071-22080.

69. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215.

70. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *IEEE International Conference on Computer Vision* (pp. 618-626).

71. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135-1144).

72. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp. 4765-4774).

73. Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint arXiv:1702.08608*.

74. Zhang, Q. S., & Zhu, S. C. (2018). Visual interpretability for deep learning: a survey. *Frontiers of Information Technology & Electronic Engineering*, 19(1), 27-39.

75. Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What clinicians want: contextualizing explainable machine learning for clinical end use. In *Machine Learning for Healthcare Conference* (pp. 359-380). PMLR.

76. Amann, J., Blasimme, A., Vayena, E., Frey, D., Madai, V. I., & Precise QC Consortium. (2020). Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC Medical Informatics and Decision Making*, 20(1), 1-9.

77. Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1721-1730).

78. Choi, E., Bahadori, M. T., Schuetz, A., Stewart, W. F., & Sun, J. (2016). Doctor AI: Predicting clinical events via recurrent neural networks. In *Machine Learning for Healthcare Conference* (pp. 301-318). PMLR.

79. Rajkomar, A., Oren, E., Chen, K., Dai, A. M., Hajaj, N., Hardt, M., ... & Dean, J. (2018). Scalable and accurate deep learning with electronic health records. *NPJ Digital Medicine*, 1(1), 18.

80. Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., & Galstyan, A. (2019). Multitask learning and benchmarking with clinical time series data. *Scientific Data*, 6(1), 96.

81. Gianfrancesco, M. A., Tamang, S., Yazdany, J., & Schmajuk, G. (2018). Potential biases in machine learning algorithms using electronic health record data. *JAMA Internal Medicine*, 178(11), 1544-1547.

82. Chen, I. Y., Johansson, F. D., & Sontag, D. (2018). Why is my classifier discriminatory? In *Advances in Neural Information Processing Systems* (pp. 3539-3550).

83. Pfohl, S. R., Duan, T., Ding, D. Y., & Shah, N. H. (2021). Counterfactual reasoning for fair clinical risk prediction. In *Machine Learning for Healthcare Conference* (pp. 325-358). PMLR.

84. Kaushal, A., Altman, R., & Langlotz, C. (2020). Geographic distribution of US cohorts used to train deep learning algorithms. *JAMA*, 324(12), 1212-1213.

85. FDA. (2021). *Artificial intelligence/machine learning (AI/ML)-based software as a medical device (SaMD) action plan*. US Food and Drug Administration.

86. Benjamens, S., Dhunnoo, P., & Meskó, B. (2020). The state of artificial intelligence-based FDA-approved medical devices and algorithms: an online database. *NPJ Digital Medicine*, 3(1), 118.

87. Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... & Liang, P. (2021). On the opportunities and risks of foundation models. *arXiv preprint arXiv:2108.07258*.

88. Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., ... & Natarajan, V. (2023). Large language models encode clinical knowledge. *Nature*, 620(7972), 172-180.

89. Moor, M., Banerjee, O., Abad, Z. S. H., Krumholz, H. M., Leskovec, J., Topol, E. J., & Rajpurkar, P. (2023). Foundation models for generalist medical artificial intelligence. *Nature*, 616(7956), 259-265.

90. Chen, Z., Varma, M., & Delbrouck, J. (2023). BiomedCLIP: A multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. *arXiv preprint arXiv:2303.00915*.
