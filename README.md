# Projet de Fin de Module — Deep Learning
### EMSI Casablanca · Filière Informatique · 2025–2026

---

## Idée générale

Ce projet explore et compare **trois grandes familles d'architectures de deep learning**, chacune appliquée à un type de données réel :

| Partie | Architecture | Dataset | Tâche |
|--------|-------------|---------|-------|
| **I** | MLP (Perceptron Multi-Couches) | Breast Cancer Wisconsin | Classification binaire (Bénin / Malin) |
| **II** | CNN (LeNet-5 Modernisé) | Digit Recognizer (MNIST) | Reconnaissance de chiffres manuscrits (0–9) |
| **III** | Seq2Seq LSTM + Attention Bahdanau | English-French Translation | Traduction automatique EN → FR |

L'objectif est de montrer que le choix d'une architecture n'est pas arbitraire : il doit être guidé par la **nature intrinsèque des données** (tabulaire, image, séquence) et les **propriétés structurelles** à exploiter.

Chaque partie couvre :
- Étude théorique (formules, propriétés)
- Implémentation PyTorch modulaire et non hard-codée (CONFIG global)
- Comparaisons expérimentales contrôlées
- Évaluation avec métriques adaptées
- Question de synthèse critique

---

## Structure du dépôt

```
deep_learning_projet/
│
├── Partie_I_MLP.ipynb               # Notebook Partie I — MLP
├── Partie_II_CNN.ipynb              # Notebook Partie II — CNN
├── Partie_III_Seq2Seq.ipynb         # Notebook Partie III — Seq2Seq
│
├── Rapport_Deep_Learning.md         # Rapport scientifique complet (Markdown)
├── Rapport_DeepLearningg.docx       # Rapport Word avec 16 graphes intégrés
├── Rapport__DeepLearning.pdf        # Rapport au format PDF
│
└── rapport_graphs/                  # Graphes matplotlib générés (PNG)
    ├── g1_class_distribution.png    # Distribution classes + corrélation features (Partie I)
    ├── g1_init_distributions.png    # Distribution poids selon stratégie d'init (Partie I)
    ├── g1_training_curves.png       # Courbes loss/accuracy MLP (Partie I)
    ├── g1_init_comparison.png       # Comparaison 3 stratégies d'initialisation (Partie I)
    ├── g1_confusion_matrix.png      # Matrice de confusion MLP (Partie I)
    ├── g2_mnist_samples.png         # Exemples MNIST par classe (Partie II)
    ├── g2_conv_kernels.png          # Effet des noyaux de convolution (Partie II)
    ├── g2_training_curves.png       # Courbes loss/accuracy LeNet-5 (Partie II)
    ├── g2_architecture_comparison.png  # Comparaison CNN vs MLP (Partie II)
    ├── g2_feature_maps.png          # Feature maps blocs 1 et 2 (Partie II)
    ├── g2_confusion_matrix.png      # Matrice de confusion LeNet-5 10 classes (Partie II)
    ├── g3_training_curves.png       # Courbes loss/perplexité Seq2Seq (Partie III)
    ├── g3_convergence_rnn.png       # Convergence comparée RNN/GRU/LSTM (Partie III)
    ├── g3_rnn_comparison.png        # PPL + BLEU + paramètres : RNN vs GRU vs LSTM (Partie III)
    ├── g3_bleu_comparison.png       # Greedy vs Beam Search BLEU (Partie III)
    └── g3_attention_heatmap.png     # Heatmap attention Bahdanau EN→FR (Partie III)
```

---

## Détail des notebooks

### `Partie_I_MLP.ipynb` — MLP et ingénierie PyTorch

- **Dataset :** Breast Cancer Wisconsin (569 exemples, 30 features numériques)
- **Architecture :** `Input(30) → [Linear→BN→ReLU→Dropout(0.3)]×3 [128,64,32] → Output(2)`
- **Implémentation :** deux versions — `nn.Sequential` et classe `MLP(nn.Module)` personnalisée
- **Expériences :** comparaison de 3 stratégies d'initialisation (Gaussienne, Constante, Xavier Uniform)
- **Résultats :** Xavier Uniform → Test Accuracy **97.7%**, F1 **0.977**
- **Extras :** sauvegarde/rechargement checkpoint, early stopping, inspection `named_parameters()`

### `Partie_II_CNN.ipynb` — CNN et vision par ordinateur

- **Dataset :** Digit Recognizer / MNIST (42 000 images 28×28, 10 classes)
- **Architecture :** LeNet-5 modernisé avec BatchNorm, ReLU, Dropout
- **Implémentation :** convolution 2D manuelle (NumPy) validée contre PyTorch (diff < 1e-5)
- **Expériences :** comparaison LeNet-5 vs SimpleCNN vs MLP-Flat
- **Résultats :** LeNet-5 → Val Accuracy **98.9%** (~61 750 paramètres)
- **Extras :** visualisation feature maps blocs 1 et 2, matrices de confusion par chiffre

### `Partie_III_Seq2Seq.ipynb` — RNN / LSTM / GRU / Seq2Seq

- **Dataset :** 30 000 paires English-French (sur ~175 000 disponibles)
- **Architecture :** Encodeur BiLSTM + Attention de Bahdanau + Décodeur LSTM
- **Expériences :** comparaison RNN Vanille vs GRU vs LSTM (10 époques, même config)
- **Décodage :** Greedy Search et Beam Search (beam_size=5)
- **Résultats :**

  | Modèle | Val PPL | BLEU-4 |
  |--------|---------|--------|
  | RNN Vanille | 22.6 | 0.089 |
  | GRU | 10.8 | 0.164 |
  | **LSTM ★** | **10.1** | **0.183** |

- **Extras :** visualisation heatmap attention, calcul BLEU via `nltk`, Beam Search +9.8% BLEU-4 vs Greedy

---

## Technologies utilisées

- **Python 3.x** · **PyTorch** · **NumPy** · **Pandas** · **Matplotlib**
- **scikit-learn** (preprocessing, métriques) · **nltk** (BLEU score)
- **torchvision** (fallback MNIST) · **python-docx** (rapport Word)

---

## Reproductibilité

Tous les notebooks fixent le seed aléatoire (`seed=42`) pour PyTorch, NumPy et Python `random`. Les hyperparamètres sont centralisés dans un dictionnaire `CONFIG` en cellule 2 de chaque notebook — il suffit de modifier ce dictionnaire pour adapter l'expérience.

Les données sont chargées automatiquement via l'API Kaggle (si configurée) avec **fallback automatique** vers les sources intégrées (`sklearn`, `torchvision`) si Kaggle n'est pas disponible (compatible Google Colab sans configuration).

---

## Résultats clés

| Partie | Modèle | Métrique principale |
|--------|--------|-------------------|
| I — MLP | Xavier Uniform | Accuracy **97.7%** · F1 **0.977** |
| II — CNN | LeNet-5 Modernisé | Val Accuracy **98.9%** |
| III — Seq2Seq | LSTM + Attention | BLEU-4 **0.183** (greedy) · **0.201** (beam b=5) |

---

*Projet réalisé dans le cadre du module Deep Learning — EMSI Casablanca 2025–2026.*
