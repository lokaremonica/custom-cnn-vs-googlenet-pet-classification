# 🐶 Pet Breed Classification: Custom CNN vs GoogleNet (Transfer Learning)

This project explores image classification on a custom dataset derived from the **Oxford-IIIT Pet Dataset**. A custom Convolutional Neural Network (CNN) is compared with **GoogleNet** to analyze the power of transfer learning.

---

## 🎯 Objective

> Classify pet images into one of three breeds: **Beagle**, **Pug**, or **Chihuahua** using both a custom-built CNN and a pretrained GoogleNet model.

---

## 📦 Dataset

* Based on `OxfordIIITPet` dataset from `torchvision.datasets`
* **Filtered 3 classes**: Beagle, Pug, Chihuahua
* **100 images each**, totaling **300 images**
* Split: **80% training / 20% test**

---

## 🧪 Data Preprocessing

* Image resized to `224x224`
* Training transforms: random horizontal flip, random rotation
* Normalized using ImageNet mean and std
* Handled using PyTorch’s `ImageFolder` and `DataLoader`

---

## 🧠 Models

### 🔹 Custom CNN

```python
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Flatten → Linear(128) → ReLU → Linear(3)
```

* **Optimizer**: Adam
* **Loss**: CrossEntropyLoss
* **Epochs**: 15

📉 **Train Accuracy**: 83.33%
📉 **Test Accuracy**: 51.67%

> ⚠️ The model learned well during training but didn’t generalize well to unseen data due to small dataset size and potential overfitting.

---

### 🔹 Transfer Learning with GoogleNet

* Used pretrained **GoogleNet** (`torchvision.models.googlenet`)
* Final `fc` layer modified: `Linear(1024 → 3)`
* **Frozen weights**, only trained the classification head
* **Epochs**: 5
* **Optimizer**: Adam
* **Loss**: CrossEntropyLoss

✅ **Test Accuracy**: **91.67%**

> 🚀 Transfer learning enabled rapid convergence and high performance on a small dataset.

---

## 📊 Performance Comparison

| Model                | Train Accuracy | Test Accuracy |
| -------------------- | -------------- | ------------- |
| Custom CNN           | 83.33%         | 51.67%        |
| GoogleNet (5 epochs) | —              | **91.67%**    |

---

## 📈 Sample Inference

> Example prediction: "Chihuahua" with 89.05% confidence

---

## 🧠 Key Learnings

* **Transfer Learning** is highly effective for small datasets
* **Custom CNN** struggles to generalize with limited training data
* Data augmentation helps but is not a substitute for model depth
* ReLU activation and CrossEntropyLoss remain effective defaults

---

## 🏁 Conclusion

This project showcases a full image classification pipeline, from custom dataset creation to performance comparison between a hand-crafted CNN and a state-of-the-art pretrained model. Transfer learning with GoogleNet dramatically outperformed the custom CNN with fewer epochs and better generalization.

---
