# Custom-CNN-with-CBAM-for-Sandfly-Genitalia-Classification

## The Question That Started It All

A simple query ignited this research, which is: When a disease outbreak happens and millions of people are at risk, what's the first thing you need to know? You need to know what you're dealing with. What species is carrying the disease? What's the population of females that are actually transmitting it?

That's exactly the problem we're tackling in our research on gender-based classification of sandfly genitalia using attention-based CNN and pre-trained models, a strategic approach to identify disease vectors quickly and accurately when seconds count.

---

## Overview

This project develops an automated, interpretable deep learning system for rapid gender and species classification of sandfly genitalia. By combining custom Convolutional Neural Networks with Convolutional Block Attention Modules (CBAM), we achieve 96-98% accuracy while maintaining explainability—critical for disease vector identification in resource-limited settings where seconds count.

## Problem Statement

Sandflies are among the most dangerous insects in the world. They're responsible for transmitting leishmaniasis, a parasitic disease that affects millions of people annually across tropical and subtropical regions. The female sandflies are the primary vectors; only females bite humans and transmit the disease.

However, traditional disease intelligence relies on manual identification. Experts must examine sandfly specimens under a microscope, look at their genitalia morphology, and classify both the species and gender. It's tedious, slow, and requires highly trained specialists who are often unavailable in resource-limited settings where the disease burden is highest.

### Critical Gaps in Current Practice

**Disease Burden**: Leishmaniasis affects millions annually across tropical regions where healthcare infrastructure is already stretched thin.

**Traditional Limitations**: Manual identification can take hours per batch of specimens, making it impossible to scale in outbreak situations.

**The Opportunity**: Modern deep learning with attention mechanisms could make classification fast, accurate, and accessible everywhere.

## Our Motivation

We needed to build a system that's not just accurate, but also interpretable and transparent—one that experts can trust because they can see why the model made each decision. That's where explainable AI comes in. We wanted to deliver:

- **Speed**: Automated classification in seconds instead of hours
- **Accuracy**: High-performance models that rival expert analysis
- **Transparency**: Clear justification for every prediction
- **Accessibility**: Solutions deployable in resource-limited settings

## Related Work

Previous studies used CNNs for insect classification with high accuracy but little explainability. Later models added attention to focus on key parts, yet still acted like black boxes. Traditional methods using wing shapes and textures were less accurate and needed manual work. In another study for malaria detection, Grad-CAM helped show what models looked at, building expert trust but not scaling well. Our work advances these approaches by combining attention mechanisms with gender-specific classification and validated explainability.

## Dataset

**Source**: High-resolution genitalia images from Mendeley

**Species** (3 medically important sandfly species):
- Phlebotomus papatasii
- Phlebotomus alexandri
- Phlebotomus sergenti

**Classes**: 6 total (male and female for each species)

### Preprocessing Pipeline

- Removed background and added black background behind to focus on the genitalia only
- Cropped images to focus on the genitalia region
- Resized to 256×256 pixels
- Applied augmentation: rotations, flips, and brightness adjustments to increase dataset variation

**Data Split**: 80% training | 10% validation | 10% testing

## Methodology

Our workflow is straightforward:

1. **Preprocessed** the dataset by removing backgrounds and applying augmentation
2. **Split** into 80/10/10 ratio for training, validation, and testing
3. **Trained** three models: Custom CNN with CBAM, MobileNetV2, and DenseNet201
4. **Evaluated** all models with different metrics (accuracy, precision, recall, F1-score)
5. **Visualized** with Grad-CAM to ensure models learn real biological features

## Model Architecture

We tested three architectures: a Custom CNN with CBAM, MobileNetV2, and DenseNet201. We focused on the Custom CNN with CBAM, as it really highlights the power of attention mechanisms.

### Custom CNN with CBAM (Primary Model)

A key part of this model is the **Convolutional Block Attention Module (CBAM)**. CBAM helps the network focus on the most important information in the input through two steps:

<img width="543" height="240" alt="image" src="https://github.com/user-attachments/assets/fe5ead70-fa7c-4edb-8e54-76cd4b3fbda8" />

**Channel Attention**: Looks at each feature channel (like color, texture, or edges) and highlights the most important ones, acting like a spotlight on the useful features.

**Spatial Attention**: Decides where to focus by creating a map of the important areas in the image, guiding the model to concentrate on key regions instead of the entire input.

In our model, we place CBAM after each convolutional layer, so the network refines features right after extraction. This makes learning more efficient, improves accuracy, and makes the process more interpretable.

**In short**: CBAM teaches the model to focus on the right features in the right places, boosting both performance and feature learning.

```
Input (256×256) 
  ↓
Conv2D + ReLU + CBAM
  ↓
Conv2D + ReLU + CBAM
  ↓
MaxPooling2D
  ↓
Conv2D + ReLU + CBAM
  ↓
Conv2D + ReLU + CBAM
  ↓
GlobalAveragePooling2D
  ↓
Dense + Dropout (60%)
  ↓
Dense (6 classes, softmax)
```

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Batch Size | 10 |
| Optimizer | Adam |
| Learning Rate | 0.0009 |
| Loss Function | Categorical Cross-Entropy |
| Dropout Rate | 60% |
| Early Stopping | Patience = 5 |

## Results

### Overall Performance

Our proposed custom model achieved 96% overall accuracy with the most stable performance. MobileNetV2 reached 97%. DenseNet201 reached 98%. All precision, recall, and F1-scores per class exceed 0.92, proving the model learns genuine diagnostic features that experts can trust.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN + CBAM | 96% | >0.92 | >0.92 | >0.92 |
| MobileNetV2 | 97% | >0.92 | >0.92 | >0.92 |
| DenseNet201 | 98% | >0.92 | >0.92 | >0.92 |

### Training Dynamics

All three models learned quickly and reached high accuracy. Custom CNN with CBAM shows some slight overfitting in the middle epochs (validation accuracy fluctuates while training stays smooth), but it recovers by the end. MobileNetV2 and DenseNet201 are more stable. Despite the slight overfitting, our CBAM model still performs well because of our aggressive dropout (60%) and early stopping, which keeps it in check.

### Competitive Advantage

Our Custom CNN + CBAM surpasses previous work that achieved 97.1% accuracy but lacked gender differentiation and attention mechanisms. We deliver both. CBAM can identify which genitalial features matter and where those features are located due to its attention components. This makes predictions interpretable and biologically meaningful.

## Explainable AI & Clinical Trust

A disease control officer won't deploy a black-box system saying "this is female Ph. papatasii" without justification. That's where Grad-CAM visualization comes in.

<img width="840" height="338" alt="image" src="https://github.com/user-attachments/assets/0d57cf56-4c6e-45b4-ad97-d24ead8faa5f" />

**Grad-CAM shows exactly which image regions the model uses for its decision via heatmaps.** When we apply this to our results, the model highlights the exact genitalial structures that trained entomologists also focus on:
- Specific genitalial morphologies
- Bristle patterns
- Anatomical landmarks

This alignment validates that our model learns genuine biological features and builds expert trust through transparency.

## Discussion

Our approach represents a significant advancement in automated disease vector identification. By combining attention mechanisms with gender-specific classification, we achieve both high accuracy and biological interpretability. The CBAM architecture allows the model to learn diagnostic features that align with expert knowledge, making it trustworthy for real-world deployment.

## Limitations

**Controlled Environment Data**: Our model was trained on clean, perfect lab images with good lighting. Real field samples are messier—different lighting, different angles, sometimes damaged.

This creates a gap between training conditions and real-world deployment scenarios. To truly serve outbreak response teams, the system must be tested and refined on field-collected specimens.

## Future Work

Moving forward, we're scaling through five key initiatives:

1. **Dataset Expansion**: Expand our dataset with field-collected images and more species to improve robustness

2. **Field Validation**: Validate on actual field specimens and develop mobile applications for practical use

3. **Advanced Learning**: Apply domain adaptation and multi-modal fusion to handle real-world variability

4. **Deployment**: Create mobile applications for field teams and clinicians

5. **Multi-species Coverage**: Extend classification to additional medically important sandfly species

## Impact & Conclusion

This work shows that deep learning with CBAM attention can accelerate sandfly identification while staying interpretable. We achieved 96-98% accuracy with gender differentiation, surpassing previous methods. Our explainable visualizations confirm the model learns genuine biological features.

**What matters most isn't accuracy, it's whether this reaches the field teams and clinicians who need it.** If it helps them identify vectors faster and saves lives, then the research succeeds.

By automating what once took hours into a process that takes seconds, we enable faster outbreak response in exactly the regions that need it most.

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **Attention Mechanism**: CBAM (Convolutional Block Attention Module)
- **Transfer Learning**: MobileNetV2, DenseNet201
- **Explainability**: Grad-CAM visualization
- **Preprocessing**: Image augmentation, background removal, normalization

---
