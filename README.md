# Adversarial-Attacks-Deep-Learning-Project
This project explores adversarial robustness of deep neural networks on a subset of the **ImageNet-1K** dataset, using a **pre-trained ResNet-34** model from PyTorch. The focus is on designing, implementing, and evaluating **pixel-wise** and **patch-based** adversarial attacks that degrade model performance while maintaining visual similarity.

## 📄 Tasks Overview

This project is organized into **five major tasks**, all contained in a single Jupyter notebook: `attacks1-4.ipynb`.

### 🔹 Task 1: **Evaluate Baseline Model Accuracy**
- Loads a pre-trained ResNet-34 model (`torchvision.models.resnet34`)
- Evaluates Top-1 and Top-5 accuracy on the provided test subset of ImageNet-1K
- Uses normalization parameters aligned with ImageNet:
  ```python
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  ```

### 🔹 Task 2: **Fast Gradient Sign Method (FGSM) Attack**
- Implements **FGSM** with ε = 0.02 (L∞ constraint)
- Creates a new set of adversarial images (“Adversarial Test Set 1”)
- Evaluates attack effectiveness (accuracy drop) and visual similarity

### 🔹 Task 3: **Projected Gradient Descent (PGD) Attack**
- Improves over FGSM by applying **multiple gradient steps**
- Constructs “Adversarial Test Set 2”
- Reports performance drop (target: ≥ 70% drop in Top-1 accuracy)

### 🔹 Task 4: **Targeted Patch Attack**
- Restricts perturbations to a **32×32 patch** (ε increased to 0.5)
- Optimizes a universal adversarial patch targeting a fixed class (e.g., toaster)
- Saves results as “Adversarial Test Set 3”

### 🔹 Task 5: **Transferability Evaluation**
- Tests adversarial examples on another model (`DenseNet-121`)
- Reports accuracy across:
  - Original dataset
  - FGSM (Set 1)
  - PGD (Set 2)
  - Patch attack (Set 3)
- Discusses transferability and implications

## ▶️ Running the Notebook

1. **Clone the repo:**
   ```bash
   git clone https://github.com/RuchitJathania/Adversarial-Attacks-Deep-Learning-Project.git
   cd Adversarial-Attacks-Deep-Learning-Project
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+, PyTorch, and torchvision installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook:**
   ```bash
   jupyter notebook attacks1-4.ipynb
   ```

4. **Datasets:**
   - Place the provided **ImageNet subset** in `./TestDataSet/`
   - The notebook will automatically generate:
     - `AdversarialTestSet1/` (FGSM)
     - `AdversarialTestSet2/` (PGD)
     - `AdversarialTestSet3/` (Patch)

## 📊 Outputs

- Visualizations of 3–5 adversarial examples per method
- Top-1 and Top-5 accuracy reports per dataset
- Patch visualizations and heatmaps (optional)
- Final comparison of all 4 datasets on both ResNet-34 and DenseNet-121

## 📌 Notes

- The notebook supports both **CPU and GPU** (MPS if on Mac).
- Git LFS is recommended for storing `.ipynb` files > 50MB.
- All adversarial attacks conform to the visual similarity and ε-bound requirements.
