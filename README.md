# Text2FaceGAN & StackGAN — Génération de visages à partir de descriptions textuelles

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![BERT](https://img.shields.io/badge/Encodeur-BERT-yellow?logo=huggingface&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CelebA-green)
![Plateforme](https://img.shields.io/badge/Plateforme-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Université](https://img.shields.io/badge/Université-Paris%20Cité-8E0032)

> **Projet Pluridisciplinaire — Master 1 Machine Learning pour la Science des Données**
> Université Paris Cité | Année 2024–2025
> Encadrante : Mme. Severine AFFELDT

---

## 👥 Auteurs

| Nom | GitHub |
|-----|--------|
| Rhofra ADAHCHOUR | [@rhofra](https://github.com/) |
| Nourelhouda DJERIOU | [@nourelhouda](https://github.com/) |
| Safae SEBAY | [@safae](https://github.com/) |
| Zara ZULFIQAR | [@zara](https://github.com/) |

---

## 📋 Description du projet

Ce projet porte sur la reproduction et l'extension de l'article **Text2FaceGAN : Face Generation from Fine-Grained Textual Descriptions** (Nasir et al., 2019).

L'objectif est de générer des visages humains réalistes à partir de descriptions textuelles telles que :
> *« The woman has oval face and wavy brown hair. She has big lips. The young attractive woman has heavy makeup. »*

Le projet s'organise en deux axes principaux :
1. **Reproduction** de Text2FaceGAN en PyTorch avec remplacement de l'encodeur Skip-Thought par BERT
2. **Extension** via l'architecture StackGAN permettant une génération en `128×128` pixels avec une qualité supérieure

---

## 📁 Structure du dépôt

```
Text2FaceGAN/
│
├── 📂 stackgan/
│   ├── stackgan_train.ipynb       # Entraînement complet StackGAN (Stage-I + Stage-II)
│   └── stackgan_eval.ipynb        # Évaluation complète (FID, cohérence, mode collapse)
│
├── 📂 text2facegan/
│   └── text2facegan.ipynb         # Reproduction Text2FaceGAN en PyTorch
│
├── 📂 assets/
│   └── ...                        # Figures et résultats illustratifs
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🏗️ Architectures

### Text2FaceGAN (Reproduction)

Architecture **DCGAN conditionnel** avec formulation **GAN-CLS** :

```
Description textuelle
        ↓
   BERT (768d)
        ↓
  Projection (256d)
        ↓
  Concaténation avec z ~ U(-1,1)^100
        ↓
  Générateur G (5.8M params)
        ↓
  Image 64×64
```

| Composant | Papier original | Notre reproduction |
|-----------|----------------|-------------------|
| Encodeur texte | Skip-Thought (4800d) | Sentence-BERT (384d) |
| Résolution | 64×64 | 64×64 |
| Epochs | 200 | 200 |
| Batch size | 64 | 64 |
| Learning rate G | 2×10⁻⁴ | 2×10⁻⁴ |
| Learning rate D | 1×10⁻⁴ | 1×10⁻⁴ |
| Label swapping | Tous les 3 batches | Tous les 3 batches |

### StackGAN (Extension)

Architecture **à deux stages progressifs** avec encodeur **BERT** :

```
Description textuelle
        ↓
   BERT (768d) + Conditional Augmentation (128d)
        ↓
  ┌─────────────────────┐
  │  Stage-I (10.25M)   │  →  Image 64×64
  └─────────────────────┘
        ↓
  ┌─────────────────────┐
  │  Stage-II (26.21M)  │  →  Image 128×128
  └─────────────────────┘
```

| Composant | Valeur |
|-----------|--------|
| Encodeur texte | BERT base-uncased (768d) |
| Conditional Augmentation | 128d |
| Stage-I résolution | 64×64 |
| Stage-II résolution | 128×128 |
| Epochs Stage-I | 150 |
| Epochs Stage-II | 60 |
| Optimiseur | Adam (β₁=0.5) |

---

## 📊 Résultats

### Métriques quantitatives

| Modèle | FID ↓ | IS | Résolution |
|--------|-------|----|------------|
| Text2FaceGAN (papier) | N/R | 1.4 ± 0.7 | 64×64 |
| Text2FaceGAN (nôtre) | 138.99 | ~1.4 | 64×64 |
| **StackGAN Stage-I** | **74.21** | N/A | **64×64** |
| **StackGAN Stage-II** | **79.72** | N/A | **128×128** |

> ✅ **StackGAN réduit le FID de 46%** par rapport à la reproduction de Text2FaceGAN.

### Absence de mode collapse

La vérification sur 16 images générées pour la même description confirme l'absence de mode collapse :
- **Variance pixel** : 0.0224 (> 0.01 ✅)
- **Distance L2 moyenne** : 45.61

### Cohérence textuelle

| Attribut testé | Résultat |
|----------------|----------|
| Couleur de cheveux | ✅ Bien capturé |
| Genre (homme/femme) | ✅ Bien capturé |
| Style de cheveux | ⚠️ Partiellement capturé |
| Accessoires (lunettes, chapeau) | ❌ Difficile à reproduire |

---

## 🗃️ Dataset

**CelebA** — Large-scale CelebFaces Attributes Dataset

- 202 599 images de visages de célébrités
- 40 attributs binaires par image
- Captions générées automatiquement à partir des attributs (6 parties structurées)

| Dataset | Images utilisées | Résolution |
|---------|-----------------|------------|
| Text2FaceGAN | 10 000 | 64×64 |
| StackGAN | 10 000 | 64×64 / 128×128 |

Accès via Kaggle : [`jessicali9530/celeba-dataset`](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-repo/Text2FaceGAN.git
cd Text2FaceGAN

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Sur Kaggle (recommandé — GPU T4 gratuit)

1. Créer un nouveau notebook Kaggle
2. Ajouter le dataset CelebA via **Add Data** → `jessicali9530/celeba-dataset`
3. Uploader et exécuter `stackgan/stackgan_train.ipynb`
4. Pour l'évaluation, ajouter l'output du notebook précédent puis exécuter `stackgan/stackgan_eval.ipynb`

### Ordre d'exécution

```
1. stackgan_train.ipynb   →  Entraîne Stage-I (150 epochs) + Stage-II (60 epochs)
                              Sauvegarde : checkpoints/, bert_embeddings.pkl, captions.pkl

2. stackgan_eval.ipynb    →  Charge les checkpoints
                              Calcule FID, LPIPS, cohérence textuelle, mode collapse
```

---

## 📦 Dépendances principales

| Package | Version | Usage |
|---------|---------|-------|
| `torch` | ≥ 2.0 | Framework deep learning |
| `torchvision` | ≥ 0.15 | Transforms, métriques |
| `transformers` | ≥ 4.30 | BERT (HuggingFace) |
| `pytorch-fid` | ≥ 0.3 | Calcul FID |
| `matplotlib` | ≥ 3.7 | Visualisations |
| `Pillow` | ≥ 9.5 | Traitement images |

---

## 📚 Références

```bibtex
@article{nasir2019text2facegan,
  title   = {Text2FaceGAN: Face Generation from Fine Grained Textual Descriptions},
  author  = {Nasir, O.R. and others},
  journal = {arXiv:1911.11378},
  year    = {2019}
}

@inproceedings{zhang2017stackgan,
  title     = {StackGAN: Text to Photo-realistic Image Synthesis with Stacked GANs},
  author    = {Zhang, Han and others},
  booktitle = {ICCV},
  year      = {2017}
}

@inproceedings{devlin2018bert,
  title     = {BERT: Pre-training of Deep Bidirectional Transformers},
  author    = {Devlin, Jacob and others},
  booktitle = {NAACL},
  year      = {2019}
}

@inproceedings{liu2015celeba,
  title     = {Deep Learning Face Attributes in the Wild},
  author    = {Liu, Ziwei and others},
  booktitle = {ICCV},
  year      = {2015}
}
```

---

## 📝 Rapport

Le rapport complet du projet est disponible dans ce dépôt : [`rapport_Text2FaceGAN.pdf`](./rapport_Text2FaceGAN.pdf)

---

<div align="center">
  <sub>Université Paris Cité — Master 1 MLSD — 2025/2026</sub>
</div>
