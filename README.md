<div align="center">

# 🎭 Face Mask Detection
### *3-Class Image Classification with Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> Sistem deteksi masker wajah berbasis **Deep Learning** yang mampu mengklasifikasikan tiga kondisi:  
> **Memakai masker** · **Tidak memakai masker** · **Masker dipakai tidak benar**

<br>

```
✅  Memakai Masker          →  with_mask
❌  Tidak Memakai Masker    →  without_mask
⚠️  Masker Tidak Benar      →  mask_weared_incorrect
```

</div>

---

## 📋 Daftar Isi

- [📊 Dataset](#-dataset)
- [🏗️ Arsitektur Model](#️-arsitektur-model)
- [⚙️ Pipeline Training](#️-pipeline-training)
- [📈 Hasil & Performa](#-hasil--performa)
- [🚀 Cara Menjalankan](#-cara-menjalankan)
- [📁 Struktur Proyek](#-struktur-proyek)
- [🔧 Teknologi](#-teknologi)

---

## 📊 Dataset

<div align="center">

| Kelas | Jumlah Gambar | Label |
|:------|:---:|:------|
| 😷 Memakai Masker | **2,994** | `with_mask` |
| 😶 Tidak Memakai Masker | **2,994** | `without_mask` |
| 🙃 Masker Tidak Benar | **2,994** | `mask_weared_incorrect` |
| **Total** | **8,982** | **3 Kelas** |

</div>

```
📁 dataset/Dataset/
├── 😷 with_mask/               → 2,994 gambar
├── 😶 without_mask/            → 2,994 gambar
└── 🙃 mask_weared_incorrect/   → 2,994 gambar
```

**Sumber:** [`vijaykumar1799/face-mask-detection`](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) via Kaggle

### 🔀 Pembagian Data

```
┌─────────────────────────────────────────────────┐
│  Total: 8,982 gambar  (per kelas seimbang ✅)   │
├─────────────────────────────────────────────────┤
│  Train      │  80%  │  7,186 gambar             │
│  Validation │  20%  │  1,796 gambar             │
└─────────────────────────────────────────────────┘
```

### 🔄 Data Augmentation

```python
ImageDataGenerator(
    rescale           = 1./255,
    horizontal_flip   = True,
    rotation_range    = 15,
    zoom_range        = 0.15,
    width_shift_range = 0.10,
    height_shift_range= 0.10,
    brightness_range  = [0.8, 1.2],
    fill_mode         = 'nearest'
)
```

> ⚠️ Augmentasi **hanya** diterapkan pada data training. Validation menggunakan `rescale` saja.

---

## 🏗️ Arsitektur Model

Proyek ini mengimplementasikan **dua model** yang berbeda:

### 1️⃣ CNN Custom

```
Input (128×128×3)
        │
   ┌────▼────┐
   │ Conv2D  │ 32 filters, 3×3, ReLU
   │ BN + MP │ BatchNorm + MaxPool 2×2
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │ 64 filters, 3×3, ReLU
   │ BN + MP │ BatchNorm + MaxPool 2×2
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │ 128 filters, 3×3, ReLU
   │ BN + MP │ BatchNorm + MaxPool 2×2
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │ 256 filters, 3×3, ReLU
   │ BN + MP │ BatchNorm + MaxPool 2×2
   └────┬────┘
        │
     Flatten
        │
   Dense(256) + BN + Dropout(0.5)
        │
   Dense(128) + Dropout(0.3)
        │
  Dense(3, softmax)
        │
     Output
```

**Fitur CNN Custom:**
- ✅ BatchNormalization setelah setiap Conv block
- ✅ L2 Regularization pada Dense layers
- ✅ Double Dropout untuk mencegah overfitting
- ✅ 4 blok konvolusi bertingkat

---

### 2️⃣ Transfer Learning — MobileNetV2

```
Input (224×224×3)
        │
┌───────▼────────┐
│  MobileNetV2   │ Pre-trained on ImageNet
│  (Frozen Base) │ 154 layers
└───────┬────────┘
        │
  GlobalAveragePooling2D
        │
  Dense(256) + BN + Dropout(0.5)
        │
  Dense(128) + L2(1e-4)
        │
  Dense(3, softmax)
        │
      Output
```

**Strategi 2-Fase Fine-tuning:**

| Fase | Layer Status | Learning Rate | Epochs |
|:----:|:-------------|:---:|:---:|
| **Fase 1** | Base *frozen* — hanya top layers | `1e-3` | 25 |
| **Fase 2** | Unfreeze **50 layer terakhir** | `1e-5` | 20 |

---

## ⚙️ Pipeline Training

### 🔧 Callbacks yang Digunakan

```python
EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True   # ← otomatis ambil model terbaik
)

ModelCheckpoint(
    monitor='val_accuracy',
    save_best_only=True         # ← hanya simpan epoch terbaik
)

ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6                 # ← kurangi LR saat stuck
)
```

### ⚖️ Class Weights

Dataset seimbang (2,994 gambar/kelas), sehingga class weight = `1.0` untuk semua kelas.

---

## 📈 Hasil & Performa

### 🏆 Ringkasan Performa

<div align="center">

| Model | Val Accuracy | Status |
|:------|:---:|:---:|
| 🧠 CNN Custom | **99.22%** | ✅ Tercapai |
| 🚀 MobileNetV2 Fine-tuned | **99.11%** | ✅ Tercapai |

</div>

### 📊 Detail Hasil CNN Custom

```
=======================================================
📊 Evaluasi: CNN Custom
=======================================================
Accuracy    : 0.9922  ✅

Classification Report:
─────────────────────────────────────────────────────
                          precision  recall  f1-score
─────────────────────────────────────────────────────
mask_weared_incorrect       0.99      0.99      0.99
with_mask                   1.00      0.99      0.99
without_mask                0.99      1.00      0.99
─────────────────────────────────────────────────────
accuracy                                       0.9922
─────────────────────────────────────────────────────
```

### 📊 Detail Hasil MobileNetV2

```
=======================================================
📊 Evaluasi: MobileNetV2 Fine-tuned
=======================================================
Accuracy    : 0.9911  ✅

Classification Report:
─────────────────────────────────────────────────────
                          precision  recall  f1-score
─────────────────────────────────────────────────────
mask_weared_incorrect       0.99      0.98      0.99
with_mask                   0.99      1.00      0.99
without_mask                0.99      0.99      0.99
─────────────────────────────────────────────────────
accuracy                                       0.9911
─────────────────────────────────────────────────────
```

### 📉 Training History

> *Grafik training dihasilkan otomatis oleh notebook untuk setiap model.*

```
CNN Custom Training
──────────────────
Epoch 1  │ Train Acc: 77.85% │ Val Acc: 45.65%
Epoch ... │      ...          │     ...
Best     │ Train Acc: ~99%+  │ Val Acc: 99.22% ✅

MobileNetV2 Fase 1
──────────────────
Epoch 1  │ Train Acc: 79.95% │ Val Acc: 88.85%
Best     │ ~99%+             │ ~99%+

MobileNetV2 Fase 2 (Fine-tuning)
─────────────────────────────────
Epoch 1  │ Train Acc: 86.94% │ Val Acc: 90.80%
Best     │ ~99%+             │ 99.11% ✅
```

### 🎯 Perbandingan Akhir

```
                   Accuracy
CNN Custom      ██████████████████████ 99.22%
MobileNetV2     ██████████████████████ 99.11%
                                      │
                              Target 90% ────┤
```

> **Kedua model melampaui target akurasi 90%** 🎉

---

## 🔍 Inference — Prediksi Gambar

```python
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

def predict_image(img_path, model, img_size=224):
    img  = keras_image.load_img(img_path, target_size=(img_size, img_size))
    arr  = keras_image.img_to_array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    
    probs    = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(probs)
    label    = CLASS_LABELS[pred_idx]
    conf     = probs[pred_idx]
    
    return label, conf

# Contoh penggunaan
label, conf = predict_image("foto_saya.jpg", tl_model)
print(f"Prediksi: {label} ({conf:.2%})")
# Output: Prediksi: with_mask (96.62%)
```

---

## 🚀 Cara Menjalankan

### Prerequisites

```bash
# Library yang dibutuhkan
pip install tensorflow scikit-learn matplotlib seaborn numpy pillow
```

### Langkah-langkah

**1. Clone repository**
```bash
git clone https://github.com/username/face-mask-detection.git
cd face-mask-detection
```

**2. Buka di Google Colab**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

> Pastikan runtime di-set ke **T4 GPU**:  
> `Runtime → Change runtime type → T4 GPU`

**3. Download Dataset dari Kaggle**
```bash
# Upload kaggle.json terlebih dahulu
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d vijaykumar1799/face-mask-detection
unzip -q face-mask-detection.zip -d dataset
```

> 📌 Dapatkan `kaggle.json` dari: **Kaggle → Account → API → Create New Token**

**4. Jalankan notebook sel per sel**

```
1️⃣  Install & Import Library
2️⃣  Download Dataset
3️⃣  Konfigurasi & Data Generator
4️⃣  Helper Functions
5️⃣  Train CNN Custom
6️⃣  Train MobileNetV2
7️⃣  Evaluasi & Perbandingan
8️⃣  Inference (prediksi gambar baru)
```

**5. Download model yang sudah dilatih**

Model akan otomatis didownload setelah training selesai:
- `mobilenetv2_facemask_final.keras` (~27.5 MB)
- `cnn_facemask_final.keras` (~53 MB)

---

## 📁 Struktur Proyek

```
📦 face-mask-detection/
├── 📓 face_mask_detection.ipynb    ← Notebook utama
├── 📄 README.md                    ← Dokumentasi ini
├── 📁 dataset/                     ← (dibuat saat runtime)
│   └── 📁 Dataset/
│       ├── 😷 with_mask/
│       ├── 😶 without_mask/
│       └── 🙃 mask_weared_incorrect/
└── 📁 models/                      ← (dihasilkan saat training)
    ├── 🤖 mobilenetv2_facemask_final.keras
    ├── 🤖 cnn_facemask_final.keras
    ├── 🔒 best_cnn.keras
    ├── 🔒 best_tl_p1.keras
    └── 🔒 best_tl_finetuned.keras
```

---

## 🔧 Teknologi

<div align="center">

| Teknologi | Versi | Fungsi |
|:----------|:------|:-------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.10 | Bahasa pemrograman |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | 2.20 | Deep learning framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | built-in | Model building API |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | latest | Operasi array |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?logo=python&logoColor=white) | latest | Visualisasi grafik |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?logo=python&logoColor=white) | latest | Heatmap & visualisasi |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | latest | Metrics & evaluation |
| ![Colab](https://img.shields.io/badge/-Google_Colab-F9AB00?logo=googlecolab&logoColor=white) | T4 GPU | Runtime cloud |
| ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?logo=kaggle&logoColor=white) | — | Dataset source |

</div>

---

## 💡 Poin Teknis Penting

<details>
<summary><b>🐛 Bug yang Diperbaiki dari Versi Awal</b></summary>

```
❌ SEBELUM (val accuracy ~49% / random)
   Generator tidak di-reset sebelum predict
   → urutan y_pred tidak cocok dengan y_true

✅ SESUDAH
   generator.reset() dipanggil sebelum model.predict()
   shuffle=False pada validation generator
   → akurasi evaluasi valid dan akurat
```

</details>

<details>
<summary><b>⚙️ Teknik yang Diterapkan</b></summary>

| Teknik | Tujuan |
|:-------|:-------|
| BatchNormalization | Stabilkan training, cegah vanishing gradient |
| L2 Regularization | Cegah overfitting pada Dense layers |
| Data Augmentation | Variasi data agar model lebih generalis |
| EarlyStopping | Hentikan training saat tidak ada perbaikan |
| ReduceLROnPlateau | Turunkan learning rate saat loss stagnan |
| ModelCheckpoint | Simpan otomatis model dengan akurasi terbaik |
| 2-Fase Fine-tuning | Hindari kerusakan pretrained weights MobileNetV2 |
| `shuffle=False` di val | Pastikan y_true dan y_pred sinkron |

</details>

<details>
<summary><b>🧠 Mengapa MobileNetV2?</b></summary>

- **Ringan & efisien** — dirancang untuk perangkat mobile
- **Pre-trained ImageNet** — sudah "mengerti" fitur visual umum
- **Depthwise Separable Convolutions** — akurasi tinggi dengan parameter lebih sedikit
- **Cocok untuk transfer learning** — arsitektur modular yang mudah di-fine-tune

</details>

---

## 📌 Catatan

- Training dilakukan di **Google Colab** dengan GPU **NVIDIA T4**
- Dataset **seimbang** (2,994 gambar per kelas), sehingga tidak diperlukan oversampling
- Model terbaik disimpan otomatis via `ModelCheckpoint` dan `restore_best_weights=True`
- Threshold prediksi default: **argmax(softmax)** untuk 3 kelas

---

<div align="center">

**Made with ❤️ using TensorFlow & Google Colab**

*Face Mask Detection · Deep Learning · Computer Vision*

</div>
