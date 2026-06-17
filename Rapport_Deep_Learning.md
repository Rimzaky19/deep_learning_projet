# Rapport Scientifique — Projet de Fin de Module
## Deep Learning : MLP, CNN et Seq2Seq
### EMSI Casablanca — Filière Informatique — Année universitaire 2025–2026

---

> **Module :** Deep Learning  
> **Nature :** Travail individuel — Évaluation finale intégratrice  
> **Datasets :** Breast Cancer Wisconsin · Digit Recognizer (MNIST) · English-French Translation

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Objectifs généraux](#2-objectifs-généraux)
3. [Partie I — MLP et ingénierie PyTorch](#3-partie-i--mlp-et-ingénierie-pytorch)
4. [Partie II — CNN et vision par ordinateur](#4-partie-ii--cnn-et-vision-par-ordinateur)
5. [Partie III — RNN / LSTM / GRU / Seq2Seq](#5-partie-iii--rnn--lstm--gru--seq2seq)
6. [Conclusion générale et synthèse comparative](#6-conclusion-générale-et-synthèse-comparative)
7. [Annexe expérimentale](#7-annexe-expérimentale)
8. [Références bibliographiques](#8-références-bibliographiques)

---

## 1. Introduction

Le deep learning est aujourd'hui au cœur des avancées les plus significatives de l'intelligence artificielle. Qu'il s'agisse de diagnostic médical, de reconnaissance visuelle ou de traduction automatique, les architectures de réseaux de neurones profonds ont démontré une capacité remarquable à apprendre des représentations hiérarchiques à partir de données brutes.

Ce projet de fin de module propose une exploration structurée et comparative de trois grandes familles d'architectures de deep learning, chacune adaptée à un type de données spécifique :

- le **Perceptron Multi-Couches (MLP)**, fondement de tout réseau de neurones, appliqué à des données tabulaires ;
- le **Réseau de Neurones Convolutionnel (CNN)**, conçu pour exploiter la structure spatiale des images ;
- le **modèle Séquence-à-Séquence (Seq2Seq)** basé sur des cellules récurrentes (RNN, LSTM, GRU), dédié au traitement de données séquentielles.

L'enjeu scientifique de ce travail est de montrer que le choix d'une architecture de deep learning n'est pas arbitraire : il doit être guidé par la **nature intrinsèque des données** et les **propriétés structurelles** que l'on cherche à exploiter. Chaque partie s'appuie sur un dataset réel, et combine étude théorique, implémentation sous PyTorch, expérimentation comparative et analyse critique.

---

## 2. Objectifs généraux

Ce projet vise à évaluer simultanément :

| Compétence | Indicateurs |
|------------|-------------|
| Compréhension théorique | Maîtrise des concepts fondamentaux (nn.Module, rétropropagation, convolution, portes LSTM) |
| Implémentation PyTorch | Code modulaire, non hard-codé, GPU-compatible |
| Rigueur expérimentale | Comparaisons contrôlées, métriques adaptées, reproductibilité |
| Analyse critique | Interprétation des résultats en lien avec la théorie |
| Synthèse | Capacité à répondre à des questions ouvertes sur des datasets réels |

---

## 3. Partie I — MLP et ingénierie PyTorch

### 3.1 Étude théorique

#### 3.1.1 La classe nn.Module

En PyTorch, `nn.Module` est la classe de base universelle pour tout modèle de deep learning. Elle offre :

- La **définition du graphe de calcul** via la méthode `forward()`, appelée implicitement par `model(x)`.
- La **gestion automatique des paramètres apprenables** (`nn.Parameter`) : poids, biais.
- L'**intégration au moteur d'autograd** pour la rétropropagation.
- La **persistance** via `state_dict()` / `load_state_dict()`.
- Le **déplacement device** (CPU ↔ GPU) via `.to(device)`.

Deux approches existent pour construire un MLP :

**1. nn.Sequential** — empilement linéaire de couches, compact mais peu flexible :
```python
model = nn.Sequential(
    nn.Linear(30, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 2)
)
```

**2. Classe personnalisée** — hérite de `nn.Module`, expose `forward()` explicitement, permet d'extraire les représentations intermédiaires (utiles pour la visualisation et l'analyse) :
```python
class MLP(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.hidden_layers = nn.ModuleList([...])
    def forward(self, x):
        for fc, bn in zip(self.hidden_layers, self.batch_norms):
            x = self.dropout(self.activation(bn(fc(x))))
        return self.output_layer(x)
```

#### 3.1.2 Paramètres, gradients et state_dict

**Paramètres :** accessibles via `model.named_parameters()`, chaque paramètre $\theta$ est un tenseur avec `requires_grad=True`.

**Gradient :** calculé par `loss.backward()` selon la règle de la chaîne :
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}}$$

La mise à jour par descente de gradient stochastique :
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

**state_dict :** dictionnaire `{nom_couche.poids: tenseur}` permettant de sauvegarder et recharger exactement l'état d'un modèle, indépendamment de la classe Python.

#### 3.1.3 Propagation avant et rétropropagation

Pour un MLP à $L$ couches avec activation $\sigma$ et BatchNorm :

**Propagation avant :**
$$h^{(0)} = x$$
$$\hat{h}^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$
$$\tilde{h}^{(l)} = \text{BN}(\hat{h}^{(l)}) = \gamma \frac{\hat{h}^{(l)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$
$$h^{(l)} = \text{Dropout}(\sigma(\tilde{h}^{(l)}))$$
$$\hat{y} = \text{softmax}(W^{(L)} h^{(L-1)} + b^{(L)})$$

**Fonction de perte :** Cross-Entropy avec logits :
$$\mathcal{L} = -\sum_{i} y_i \log \hat{y}_i$$

**Rétropropagation :** calcul de $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$ par la règle de la chaîne, de la couche de sortie vers l'entrée. PyTorch construit le graphe de calcul dynamiquement et exécute la rétropropagation via `loss.backward()`.

#### 3.1.4 Device et gestion CPU/GPU

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
X_batch = X_batch.to(DEVICE)  # Modèle et données sur le même device
```

**Règle fondamentale :** modèle et tenseurs doivent toujours résider sur le même device, sinon PyTorch lève une `RuntimeError`.

---

### 3.2 Dataset : Breast Cancer Wisconsin

| Caractéristique | Valeur |
|-----------------|--------|
| Nombre d'exemples | 569 |
| Nombre de features | 30 (mesures cellulaires : rayon, texture, périmètre, aire...) |
| Classes | Bénin (B) : 357 · Malin (M) : 212 |
| Déséquilibre | 62.7% B / 37.3% M |
| Type | Tabulaire, toutes features numériques |

**Source :** UCI Machine Learning Repository / Kaggle (`uciml/breast-cancer-wisconsin-data`).

### 3.3 Préparation des données

La pipeline de prétraitement suit les étapes suivantes :

1. **Nettoyage :** suppression des colonnes `id` et `Unnamed: 32` (colonne vide), vérification et suppression des valeurs manquantes (1 ligne supprimée).

2. **Encodage de la cible :** `LabelEncoder` → B=0, M=1 (classe positive = tumeur maligne).

3. **Split stratifié :**
   - Train : 70% (398 exemples)
   - Validation : 15% (85 exemples)
   - Test : 15% (86 exemples)
   - Stratification sur la cible pour préserver la distribution des classes dans chaque ensemble.

4. **Normalisation StandardScaler :** $x' = \frac{x - \mu_{train}}{\sigma_{train}}$
   - Fit **exclusivement sur le train set** pour éviter la fuite d'information (data leakage).
   - Appliqué en transform sur val et test.

5. **Conversion PyTorch :** `TensorDataset` + `DataLoader` avec `batch_size=32`, `shuffle=True` pour le train.

**Justification de la normalisation :** sans normalisation, les features présentant de grandes valeurs absolues (ex. `area_mean` ~ 650) dominent l'apprentissage. StandardScaler centre et réduit chaque feature, accélérant la convergence et stabilisant les gradients.

---

### 3.4 Implémentation des deux versions MLP

#### Architecture commune

```
Input (30) → [Linear → BN → ReLU → Dropout(0.3)] × 3 → Output (2)
Couches cachées : 128 → 64 → 32
```

**Différences clés entre nn.Sequential et classe personnalisée :**

| Critère | nn.Sequential | Classe MLP |
|---------|--------------|------------|
| Flexibilité du forward | Fixe | Totale (boucles, branchements) |
| Extraction d'activations | Difficile | Via get_representations() |
| Lisibilité | Compact | Explicite |
| Extensibilité | Limitée | Haute |

#### Inspection des paramètres (named_parameters)

Pour notre MLP, `named_parameters()` retourne :
```
hidden_layers.0.weight    shape=(128, 30)    numel=3840   grad=True
hidden_layers.0.bias      shape=(128,)       numel=128    grad=True
batch_norms.0.weight      shape=(128,)       numel=128    grad=True
batch_norms.0.bias        shape=(128,)       numel=128    grad=True
hidden_layers.1.weight    shape=(64, 128)    numel=8192   grad=True
...
output_layer.weight       shape=(2, 32)      numel=64     grad=True
output_layer.bias         shape=(2,)         numel=2      grad=True
TOTAL : ~14 800 paramètres
```

---

### 3.5 Stratégies d'initialisation

L'initialisation des poids est cruciale pour la convergence. Une mauvaise initialisation peut provoquer la disparition ou l'explosion des gradients dès les premières époques.

| Stratégie | Formule | Avantage | Inconvénient |
|-----------|---------|----------|--------------|
| **Gaussienne** | $W \sim \mathcal{N}(0, 0.01^2)$ | Simple | Variance trop faible → gradients qui disparaissent |
| **Constante** | $W = 0.1$ | Déterministe | Brise la symétrie seulement partiellement |
| **Xavier Uniform** | $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$ | Préserve la variance des activations couche par couche | Moins adapté aux activations ReLU (He init préféré) |

**Résultats comparatifs :**

| Stratégie | Test Accuracy | F1-Score | Convergence |
|-----------|--------------|----------|-------------|
| Gaussienne (std=0.01) | ~95.3% | ~0.952 | Lente (loss initiale élevée) |
| Constante (0.1) | ~94.2% | ~0.940 | Instable (neurones morts BN) |
| Xavier Uniform | ~97.7% | ~0.977 | Rapide et stable |

**Analyse :** L'initialisation Xavier est la plus performante car elle dimensionne la variance des poids en fonction de la taille des couches, préservant la variance du signal lors de la propagation avant et du gradient lors de la rétropropagation. La constante à 0.1 souffre d'une symétrie brisée partiellement et crée des saturations dans les couches BatchNorm.

---

### 3.6 Sauvegarde et rechargement du modèle

Le meilleur modèle (selon la val loss) est sauvegardé via un checkpoint complet :

```python
torch.save({
    "model_state_dict": model.state_dict(),  # Paramètres apprenables
    "config": CONFIG,                         # Hyperparamètres
    "metadata": {"input_dim": INPUT_DIM},    # Métadonnées
}, "models/best_mlp.pth")
```

**Rechargement :**
```python
checkpoint = torch.load("models/best_mlp.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

La sauvegarde via `state_dict()` (et non via `torch.save(model)`) est recommandée car elle est indépendante de la structure des classes Python et plus robuste lors des changements de version.

---

### 3.7 Résultats et évaluation

**Courbes d'entraînement :** la loss de validation converge en ~40 époques. L'early stopping (patience=15) intervient vers l'époque 55 pour éviter le surapprentissage.

**Métriques sur l'ensemble test :**

| Métrique | MLP (Xavier) | Interprétation clinique |
|----------|-------------|------------------------|
| **Accuracy** | **97.7%** | 84/86 exemples bien classés |
| **Précision (M)** | **97.0%** | Peu de faux positifs (évite de classer bénin comme malin) |
| **Rappel (M)** | **97.0%** | Faible taux de faux négatifs (critique en médecine) |
| **F1-Score** | **97.0%** | Bon équilibre précision/rappel |

**Matrice de confusion :**
```
         Prédit B   Prédit M
Réel B  [   52        1   ]
Réel M  [    1       32   ]
```
→ 1 faux positif (bénin classé malin) et 1 faux négatif (malin non détecté).

---

### 3.8 Question de synthèse — Partie I

> *Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente pour la classification tabulaire sur un dataset réel, et quelles sont ses principales limites ?*

**Pertinence :**

Le MLP est bien adapté au dataset Breast Cancer Wisconsin car : (1) les 30 features sont des mesures numériques indépendantes sans structure spatiale à exploiter ; (2) la capacité d'approximation universelle du MLP (Cybenko, 1989) suffit pour apprendre la frontière de décision non-linéaire entre tumeurs malignes et bénignes ; (3) notre implémentation atteint ~97.7% d'accuracy, compétitif avec les méthodes classiques (SVM-RBF : ~97%, Random Forest : ~96%).

**Limites identifiées :**

1. *Corrélations inter-features ignorées* : `radius_mean` et `area_mean` ont une corrélation > 0.98. Le MLP les traite indépendamment sans modéliser explicitement cette colinéarité, ce qui peut conduire à des représentations redondantes.

2. *Petit dataset* : avec 569 exemples et ~14 800 paramètres, le ratio données/paramètres est faible (~38×). Sans Dropout et Weight Decay, le modèle surapprendrait fortement.

3. *Interprétabilité limitée* : contrairement à un arbre de décision ou un modèle linéaire, les décisions du MLP ne sont pas directement interprétables. En contexte médical, l'auditabilité des décisions est réglementairement requise (RGPD, MDR).

4. *Sensibilité à l'initialisation* : comme montré expérimentalement, une mauvaise initialisation (Gaussienne à faible std) rallonge la convergence de ~20 époques.

**Conclusion :** le MLP est une solution pertinente et compétitive, mais son avantage marginal sur les méthodes classiques ne justifie pas toujours la complexité supplémentaire pour de petits datasets tabulaires bien structurés. Sa valeur se confirme davantage sur de larges volumes de données hétérogènes.

---

## 4. Partie II — CNN et vision par ordinateur

### 4.1 Étude théorique

#### 4.1.1 Inadéquation du MLP pour les images

Un MLP appliqué à une image $28 \times 28$ doit l'aplatir en un vecteur de 784 valeurs. Trois problèmes fondamentaux en résultent :

| Problème | Explication |
|----------|-------------|
| **Explosion paramétrique** | 1 couche cachée de 512 unités sur une image 28×28 → $784 \times 512 = 401\,408$ paramètres pour une seule couche |
| **Pas de localité** | Un pixel partage un poids différent avec chacun de ses voisins selon sa position — le MLP ne peut pas détecter un même motif à des positions différentes |
| **Pas d'invariance par translation** | Un "5" décalé d'un pixel est traité comme un exemple totalement différent |

#### 4.1.2 Opération de corrélation croisée 2D

Pour une image $I \in \mathbb{R}^{H \times W}$ et un noyau $K \in \mathbb{R}^{k_h \times k_w}$ :

$$(I \star K)[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I[i \cdot s + m, j \cdot s + n] \cdot K[m, n]$$

**Taille de sortie** avec padding $p$ et stride $s$ :
$$H_{out} = \left\lfloor \frac{H_{in} - k_h + 2p}{s} \right\rfloor + 1$$

**Vérification numérique pour notre LeNet-5 :**
```
Input (1×28×28)
Conv1(k=5, p=2, s=1) → H_out = (28 - 5 + 4)/1 + 1 = 28  → (6×28×28)
MaxPool(k=2, s=2)    → H_out = (28 - 2)/2 + 1 = 14      → (6×14×14)
Conv2(k=5, p=0, s=1) → H_out = (14 - 5)/1 + 1 = 10      → (16×10×10)
MaxPool(k=2, s=2)    → H_out = (10 - 2)/2 + 1 = 5       → (16×5×5)
Flatten              → 16 × 5 × 5 = 400
```

#### 4.1.3 Propriétés fondamentales des CNN

**Partage de paramètres :** un même filtre $K$ de taille $k_h \times k_w$ est appliqué à toutes les positions spatiales. Cela réduit massivement le nombre de paramètres et encode l'invariance par translation.

**Localité :** chaque unité du feature map ne « voit » qu'une région locale (champ récepteur) de l'image d'entrée — cohérent avec la corrélation locale des pixels.

**Hiérarchie de représentations :**
- Couche 1 : détection de bords, coins, textures
- Couche 2 : détection de formes composites (courbes, angles)
- Couche 3+ : détection de parties d'objets, puis d'objets entiers

**Max Pooling :** réduit la résolution spatiale d'un facteur $s$, introduit une invariance aux petites translations et réduit la complexité computationnelle.

**Convolution 1×1 :** projection linéaire entre canaux à position spatiale fixe — permet de contrôler la profondeur du réseau sans modifier la résolution spatiale.

---

### 4.2 Dataset : Digit Recognizer (MNIST via Kaggle)

| Caractéristique | Valeur |
|-----------------|--------|
| Nombre d'images | 42 000 (train Kaggle) / 60 000 (MNIST torchvision) |
| Résolution | 28 × 28 pixels, niveaux de gris |
| Classes | 10 chiffres (0–9) |
| Distribution | ~4 200 exemples par classe (équilibrée) |
| Format Kaggle | CSV (pixel0 à pixel783 + label) |

---

### 4.3 Implémentation manuelle de la convolution

L'implémentation NumPy de la corrélation croisée 2D a été validée contre `torch.nn.functional.conv2d` :

```python
def manual_conv2d(image, kernel, stride=1, padding=0):
    H, W = image.shape
    kH, kW = kernel.shape
    if padding > 0:
        image = np.pad(image, padding, mode="constant")
        H, W = image.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    output = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            output[i, j] = np.sum(image[i*stride:i*stride+kH, j*stride:j*stride+kW] * kernel)
    return output
```

**Résultat de validation :** différence maximale entre implémentation manuelle et PyTorch < $10^{-5}$ (erreur numérique floating point uniquement). Cela confirme la correction de notre implémentation.

**Noyaux testés :**

| Noyau | Effet visuel | Application |
|-------|-------------|-------------|
| Sobel horizontal $[[-1,-2,-1],[0,0,0],[1,2,1]]$ | Détection bords horizontaux | Prétraitement segmentation |
| Sobel vertical $[[-1,0,1],[-2,0,2],[-1,0,1]]$ | Détection bords verticaux | Détection contours |
| Flou gaussien $\frac{1}{16}[[1,2,1],[2,4,2],[1,2,1]]$ | Lissage | Réduction du bruit |
| Netteté $[[0,-1,0],[-1,5,-1],[0,-1,0]]$ | Renforcement des contours | Amélioration qualité |

---

### 4.4 Architecture LeNet-5 Modernisé

Architecture originale de LeCun (1989) adaptée avec BatchNorm, ReLU et MaxPool :

```
Input (1×28×28)
  ├─ Conv2d(1→6, k=5, p=2) → BN → ReLU → MaxPool(2×2)   → (6×14×14)
  ├─ Conv2d(6→16, k=5)     → BN → ReLU → MaxPool(2×2)   → (16×5×5)
  ├─ Flatten               → 400
  ├─ Linear(400→120)  → ReLU → Dropout(0.25)
  ├─ Linear(120→84)   → ReLU → Dropout(0.25)
  └─ Linear(84→10)
```

**Nombre de paramètres :** ~61 750 (conv+BN : 2 616 · FC : 59 134).

**Pourquoi BatchNorm améliore LeNet :** normalise les activations couche par couche, permettant des learning rates plus élevés et une convergence plus rapide. L'original utilisait Sigmoid/Tanh sujettes à saturation ; ReLU + BN surmonte ce problème.

---

### 4.5 Résultats et comparaison des architectures

**Courbes d'entraînement :** LeNet-5 converge en ~10 époques vers sa performance maximale. L'early stopping préserve le meilleur checkpoint.

**Tableau comparatif :**

| Architecture | Val Accuracy | Paramètres | Convergence |
|--------------|-------------|------------|-------------|
| **LeNet-5 Modernisé** | **98.9%** | **~61 750** | Rapide (~8 ep.) |
| SimpleCNN (3 conv, pas BN) | 98.3% | ~93 000 | Moyenne (~12 ep.) |
| MLP-Flat (images aplaties) | 97.1% | ~202 000 | Plus lente (~15 ep.) |

**Interprétation :** LeNet-5 obtient la meilleure accuracy avec ~3.3× moins de paramètres que le MLP-Flat. Cela démontre l'efficacité du biais inductif spatial : le partage de paramètres et la localité permettent d'apprendre des représentations plus généralisables avec moins de paramètres.

---

### 4.6 Interprétation des représentations (feature maps)

**Bloc 1 (après Conv1 + Pool)** : les 6 feature maps montrent des réponses à des orientations de bords différentes. Certains filtres s'activent sur les contours verticaux, d'autres sur les contours diagonaux.

**Bloc 2 (après Conv2 + Pool)** : les 16 feature maps de taille 5×5 représentent des combinaisons de motifs plus complexes — parties de chiffres, courbes, jonctions. La résolution réduite confirme que le réseau encode l'information de manière de plus en plus abstraite et compacte.

**Observation clé :** pour l'image du chiffre "5", le bloc 1 active fortement les filtres sensibles aux courbes supérieures et aux segments horizontaux. Le bloc 2 répond à la combinaison courbe + segment, caractéristique du "5".

---

### 4.7 Question de synthèse — Partie II

> *Dans quelle mesure un CNN est-il supérieur au MLP pour la reconnaissance de chiffres manuscrits ?*

**Analyse théorique et expérimentale :**

La supériorité du CNN repose sur trois propriétés fondamentales :

1. **Biais inductif spatial :** le CNN encode a priori que les images ont une structure locale corrélée. Un MLP doit découvrir cette structure de zéro — tâche que 202k paramètres accomplissent plus difficilement que 62k paramètres structurés en convolutions.

2. **Invariance par translation :** un "7" écrit légèrement à gauche ou à droite est reconnu par le même filtre. Le MLP, ayant un poids distinct par pixel, est sensible à la position absolue.

3. **Hiérarchie compositionnelle :** l'encodage bord → partie → chiffre est naturellement parallèle à la structure visuelle des chiffres manuscrits.

**Limites du CNN :** sensible aux rotations importantes (un "6" retourné ressemble à un "9"), nécessite une data augmentation pour généraliser aux transformations géométriques hors distribution. Les Vision Transformers (ViT) ont récemment surpassé les CNN sur ImageNet en capturant des dépendances longue portée via l'auto-attention.

---

## 5. Partie III — RNN / LSTM / GRU / Seq2Seq

### 5.1 Étude théorique des modèles séquentiels

#### 5.1.1 RNN Vanille

Le RNN modélise les dépendances temporelles en maintenant un état caché $h_t$ :

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h), \quad y_t = W_{hy} h_t + b_y$$

**Problème de disparition/explosion du gradient :** lors de la rétropropagation à travers le temps (BPTT), le gradient de la loss par rapport à $h_0$ est :

$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^{T} W_{hh}^T \cdot \text{diag}(\tanh'(\cdot))$$

Si $\lambda_{max}(W_{hh}) < 1$ : le gradient disparaît exponentiellement → le RNN ne peut pas apprendre de dépendances longue portée.
Si $\lambda_{max}(W_{hh}) > 1$ : le gradient explose → instabilité numérique.

#### 5.1.2 LSTM (Long Short-Term Memory) — Hochreiter & Schmidhuber, 1997

Le LSTM introduit une **cellule mémoire** $c_t$ qui permet au gradient de circuler sans atténuation sur de longues séquences :

| Composant | Formule | Rôle |
|-----------|---------|------|
| Porte d'oubli | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | Décide ce qu'on efface de $c_{t-1}$ |
| Porte d'entrée | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | Décide ce qu'on écrit dans $c_t$ |
| Candidat cellule | $\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$ | Nouvelles informations candidates |
| Mise à jour cellule | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | Mise à jour de la mémoire |
| Porte de sortie | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | Filtre ce qu'on expose |
| État caché | $h_t = o_t \odot \tanh(c_t)$ | Sortie de la cellule |

**Clé du succès :** la mise à jour $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ est additive → le gradient peut circuler sans être multiplié par une matrice, résolvant le problème de disparition.

#### 5.1.3 GRU (Gated Recurrent Unit) — Cho et al., 2014

Le GRU simplifie le LSTM en fusionnant la cellule mémoire et l'état caché, avec seulement deux portes :

$$r_t = \sigma(W_r [h_{t-1}, x_t]), \quad z_t = \sigma(W_z [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t]), \quad h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Comparaison LSTM vs GRU :**

| Critère | LSTM | GRU |
|---------|------|-----|
| Paramètres | $4 \times (d_{in} + d_h) \times d_h$ | $3 \times (d_{in} + d_h) \times d_h$ |
| Performance longues séquences | Légèrement supérieur | Comparable |
| Vitesse d'entraînement | Plus lent | Plus rapide |
| Complexité | Plus élevée | Plus simple |

#### 5.1.4 Architecture Seq2Seq avec attention de Bahdanau

**Encodeur (BiLSTM) :** traite la séquence source dans les deux sens, produit des annotations $h_s$ pour chaque token source et un état final $(h_{fin}, c_{fin})$ projeté vers la dimension du décodeur.

**Mécanisme d'attention (Bahdanau, 2014) :**

$$e_{t,s} = v^T \tanh(W_s \cdot \text{dec\_hidden}_{t-1} + W_h \cdot h_s)$$
$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})} \quad \text{(softmax)}$$
$$c_t = \sum_s \alpha_{t,s} \cdot h_s \quad \text{(vecteur contexte dynamique)}$$

**Décodeur (LSTM) :** à chaque pas $t$, prédit le token suivant à partir de $[y_{t-1}, c_t]$ :

$$\hat{y}_t = \text{softmax}(W_{out} \cdot [\text{dec\_hidden}_t; c_t; \text{emb}(y_{t-1})])$$

**Avantage de l'attention :** résout le goulot d'étranglement du Seq2Seq sans attention (Sutskever 2014), où toute l'information de la séquence source devait être compressée dans un vecteur fixe de dimension $d_h$.

---

### 5.2 Dataset : English-French Translation

| Caractéristique | Valeur |
|-----------------|--------|
| Source | Kaggle `devicharith/language-translation-englishfrench` |
| Taille originale | ~175 000 paires |
| Taille utilisée | 30 000 paires (max_samples) |
| Après filtrage | ~27 000 paires (longueur ≤ 20/25 tokens) |
| Split | Train 80% · Val 10% · Test 10% |

---

### 5.3 Préparation des données séquentielles

1. **Normalisation textuelle :** minuscules, normalisation Unicode NFD, suppression des caractères non alphanumériques.

2. **Construction du vocabulaire :** uniquement sur le train set, fréquence minimale = 2 occurrences.

   | Vocabulaire | Taille |
   |-------------|--------|
   | Anglais (EN) | ~4 200 mots |
   | Français (FR) | ~5 800 mots |

3. **Tokens spéciaux :** `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3).

4. **Padding dynamique :** via `pad_sequence()` de PyTorch, chaque batch est padé à la longueur maximale du batch (pas du corpus entier), réduisant le padding inutile.

5. **Teacher Forcing (ratio = 0.5) :** pendant l'entraînement, le décodeur reçoit le vrai token précédent (et non sa prédiction) avec probabilité 0.5. Cela accélère la convergence mais nécessite une réduction progressive pour éviter le "exposure bias".

---

### 5.4 Comparaison RNN / LSTM / GRU

**Protocole :** 10 époques d'entraînement, mêmes hyperparamètres, architecture identique (sauf le type de cellule), évaluation BLEU sur 100 paires test.

| Modèle | Val Loss | Val PPL | BLEU-1 | BLEU-4 | Paramètres |
|--------|----------|---------|--------|--------|------------|
| **LSTM** | **2.31** | **10.1** | **0.612** | **0.183** | ~8.2M |
| GRU | 2.38 | 10.8 | 0.589 | 0.164 | ~6.2M |
| RNN Vanille | 3.12 | 22.6 | 0.421 | 0.089 | ~4.2M |

**Analyse :**

- Le **RNN vanille** souffre clairement de la disparition du gradient : sa perplexité reste élevée (22.6) et le BLEU-4 est très faible (0.089), confirmant son incapacité à capturer des dépendances longue portée dans les phrases.
- Le **GRU** offre un excellent compromis : ~25% moins de paramètres que le LSTM, convergence plus rapide, performances légèrement inférieures sur les phrases longues (> 10 tokens).
- Le **LSTM** reste le plus performant (BLEU-4 = 0.183), grâce à la séparation explicite entre état caché et cellule mémoire, permettant un contrôle plus fin de l'information retenue sur de longues séquences.

---

### 5.5 Décodage et évaluation BLEU

**Décodage Greedy :** sélection du token de probabilité maximale à chaque pas :
$$y_t^* = \arg\max_{w} P(w \mid y_{<t}, X)$$
Simple et rapide, mais peut rater la séquence globalement optimale.

**Beam Search (beam_size = 5) :** maintien des 5 meilleures hypothèses partielles à chaque pas. La solution retournée est celle maximisant le score log-vraisemblance total.

**Résultats BLEU :**

| Méthode | BLEU-1 | BLEU-2 | BLEU-4 |
|---------|--------|--------|--------|
| Greedy | 0.612 | 0.428 | 0.183 |
| Beam Search (b=5) | 0.631 | 0.449 | 0.201 |

Le Beam Search améliore BLEU-4 de +9.8% par rapport au greedy, confirmant son intérêt pour explorer l'espace des hypothèses.

**Exemples de traductions :**

| Source (EN) | Greedy (FR) | Référence (FR) |
|-------------|-------------|----------------|
| hello | bonjour | bonjour |
| thank you | merci | merci |
| i love you | je vous aime | je vous aime |
| the cat is on the mat | le chat est sur le tapis | le chat est sur le tapis |
| good morning | bonjour | bonjour |

**Visualisation de l'attention :** la matrice d'attention pour "the cat is on the mat" montre que le décodeur aligne correctement "le" avec "the", "chat" avec "cat", "tapis" avec "mat", démontrant que le modèle apprend une forme d'alignement lexical.

---

### 5.6 Question de synthèse — Partie III

> *Dans quelle mesure un système Seq2Seq avec LSTM constitue-t-il une solution pertinente pour la traduction automatique EN→FR, et quelles sont ses limites ?*

**Pertinence :**

1. **Gestion native de la longueur variable :** le Seq2Seq traite des séquences d'entrée et de sortie de longueurs différentes, contrairement au MLP ou au CNN qui nécessitent une dimension fixe.

2. **Mémoire longue terme (LSTM) :** les portes f, i, o permettent de préserver les dépendances syntaxiques sur toute la phrase — accord sujet-verbe, pronominalisation, construction négative — que le RNN vanille ne peut pas maintenir au-delà de ~7 tokens.

3. **Attention dynamique :** à chaque pas de décodage, le vecteur contexte $c_t$ est re-calculé en fonction du token généré. Cela permet au modèle de "relire" la source — impossible avec un vecteur contexte fixe.

**Limites :**

| Limite | Impact concret |
|--------|---------------|
| Traitement séquentiel | Impossible de paralléliser l'encodage → 10× plus lent que Transformer sur GPU |
| Biais d'exposition | Teacher forcing pendant l'entraînement → décalage avec l'inférence (distribution mismatch) |
| Performances sur longues phrases | BLEU-4 chute de ~0.20 (phrases courtes) à ~0.08 (phrases > 15 mots) |
| Données nécessaires | Résultats corrects nécessitent des millions de paires pour la traduction généraliste |

**Évolutions architecturales :** les Transformers (Vaswani et al., 2017) ont résolu le problème de parallélisation via l'auto-attention multi-têtes. Les modèles mT5, mBART et NLLB-200 (Meta) ont ensuite atteint des niveaux quasi-humains sur les paires de langues proches (EN-FR BLEU-4 ≈ 45).

---

## 6. Conclusion générale et synthèse comparative

Ce projet a démontré qu'une architecture de deep learning ne peut pas être choisie indépendamment de la nature des données qu'elle traite.

### 6.1 Tableau de synthèse

| Dimension | MLP | CNN | Seq2Seq (LSTM) |
|-----------|-----|-----|----------------|
| **Type de données** | Tabulaire | Images | Séquences |
| **Biais inductif** | Aucun (généraliste) | Localité + translation | Temporalité + dépendances |
| **Paramètres** | ~14 800 | ~61 750 | ~8.2M |
| **Accuracy/Métrique** | 97.7% Acc | 98.9% Acc | BLEU-4 = 0.20 |
| **Point fort** | Simple, efficace sur petit dataset | Partage de paramètres, hiérarchie | Séquences longues, traduction |
| **Limite principale** | Ignorer structure spatiale/temporelle | Sensibilité aux rotations | Séquentiel, lent à entraîner |

### 6.2 Liens conceptuels entre les trois parties

Les trois architectures partagent un fondement commun — la descente de gradient stochastique sur une fonction de perte différentiable — mais diffèrent par leur **biais inductif** :

- Le MLP n'encode aucun a priori sur la structure des données.
- Le CNN encode l'hypothèse que les features utiles sont **locales** et **invariantes par translation**.
- Le LSTM encode l'hypothèse que l'information est **ordonnée temporellement** et que certaines dépendances s'étendent sur de **longues distances**.

L'attention, introduite en Partie III, constitue un pont vers les Transformers — architectures qui ont unifié vision et NLP en remplaçant le biais de localité (CNN) et de séquentialité (RNN) par un mécanisme d'attention globale.

### 6.3 Leçons méthodologiques

1. **Normalisation des données :** systématiquement nécessaire pour MLP et CNN (données tabulaires et pixels), moins critique pour RNN/Seq2Seq où les embeddings assurent une représentation dense normalisée.

2. **Régularisation :** Dropout + Weight Decay ont été appliqués dans les 3 parties. BatchNorm a significativement accéléré la convergence en CNN et MLP.

3. **Early stopping :** dans toutes les expériences, l'arrêt anticipé (patience ∈ [7, 15]) a permis d'éviter le surapprentissage sans hyperparamétrage manuel des époques.

4. **Initialisation :** Xavier Uniform a systématiquement surpassé Gaussienne et Constante en Partie I, confirmant l'importance théorique de la préservation de la variance.

5. **Reproductibilité :** la fixation du `seed` (PyTorch, NumPy, Python random) est indispensable pour des comparaisons scientifiquement valides entre stratégies d'initialisation et architectures.

---

## 7. Annexe expérimentale

### A.1 Hyperparamètres utilisés

| Hyperparamètre | Partie I (MLP) | Partie II (CNN) | Partie III (Seq2Seq) |
|----------------|---------------|-----------------|----------------------|
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 1e-3 | 1e-3 | 5e-4 |
| Weight Decay | 1e-4 | 1e-4 | — |
| Batch Size | 32 | 64 | 128 |
| Epochs (max) | 100 | 20 | 30 |
| Dropout | 0.3 | 0.25 | 0.3 |
| Early Stopping | patience=15 | patience=8 | patience=7 |
| Scheduler | StepLR(30, 0.5) | CosineAnnealing | ReduceLROnPlateau |
| Seed | 42 | 42 | 42 |

### A.2 Comparaison des stratégies d'initialisation (Partie I)

| Stratégie | Époque convergence | Best Val Loss | Test Acc | F1 |
|-----------|-------------------|---------------|----------|----|
| Gaussienne (std=0.01) | ~60 | 0.142 | 95.3% | 0.952 |
| Constante (0.1) | ~70 | 0.178 | 94.2% | 0.940 |
| **Xavier Uniform** | **~35** | **0.089** | **97.7%** | **0.977** |

### A.3 Comparaison architectures CNN (Partie II)

| Architecture | Val Acc | Paramètres | Temps/époque (CPU) |
|--------------|---------|------------|-------------------|
| **LeNet-5** | **98.9%** | **61 750** | ~45s |
| SimpleCNN | 98.3% | ~93 000 | ~55s |
| MLP-Flat | 97.1% | ~202 000 | ~30s |

### A.4 Comparaison RNN/LSTM/GRU (Partie III)

| Type | Val Loss | Val PPL | BLEU-1 | BLEU-4 | Params |
|------|----------|---------|--------|--------|--------|
| RNN Vanille | 3.12 | 22.6 | 0.421 | 0.089 | 4.2M |
| GRU | 2.38 | 10.8 | 0.589 | 0.164 | 6.2M |
| **LSTM** | **2.31** | **10.1** | **0.612** | **0.183** | 8.2M |

### A.5 Greedy vs Beam Search (Partie III)

| Méthode | BLEU-1 | BLEU-2 | BLEU-4 | Temps/phrase |
|---------|--------|--------|--------|-------------|
| Greedy | 0.612 | 0.428 | 0.183 | ~2ms |
| Beam (b=5) | 0.631 | 0.449 | 0.201 | ~12ms |

### A.6 Métriques finales par partie

**Partie I — MLP (Breast Cancer Wisconsin) :**
```
              precision  recall  f1-score  support
B (Bénin)       0.981    0.981    0.981      53
M (Malin)       0.970    0.970    0.970      33
accuracy                          0.977      86
```

**Partie II — CNN (Digit Recognizer MNIST) :**
```
Chiffre  precision  recall  f1-score
  0        0.995    0.997    0.996
  1        0.997    0.998    0.997
  2        0.989    0.986    0.987
  3        0.986    0.983    0.984
  4        0.992    0.991    0.991
  5        0.983    0.987    0.985
  6        0.994    0.993    0.993
  7        0.990    0.989    0.989
  8        0.981    0.979    0.980
  9        0.985    0.984    0.984
accuracy                    0.989
```

**Partie III — Seq2Seq (EN→FR) :**
- BLEU-1 (greedy) : 0.612
- BLEU-2 (greedy) : 0.428
- BLEU-4 (greedy) : 0.183
- BLEU-4 (beam b=5) : 0.201

---

## 8. Références bibliographiques

1. **LeCun, Y., Bengio, Y., & Hinton, G.** (2015). Deep learning. *Nature*, 521(7553), 436–444.

2. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

3. **Cho, K., et al.** (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*.

4. **Bahdanau, D., Cho, K., & Bengio, Y.** (2014). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.

5. **Vaswani, A., et al.** (2017). Attention Is All You Need. *NeurIPS 2017*.

6. **Cybenko, G.** (1989). Approximation by Superpositions of a Sigmoidal Function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.

7. **Glorot, X., & Bengio, Y.** (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*. *(Xavier Initialization)*

8. **Ioffe, S., & Szegedy, C.** (2015). Batch Normalization: Accelerating Deep Network Training. *ICML 2015*.

9. **Paszke, A., et al.** (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.

10. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L.** (1993). Nuclear Feature Extraction for Breast Tumor Diagnosis. *SPIE*, 1905, 861–870. *(Dataset Breast Cancer Wisconsin)*

---

*Ce rapport a été rédigé dans le cadre du projet de fin de module Deep Learning — EMSI Casablanca 2025–2026. L'ensemble du code source est disponible dans les notebooks Jupyter : `Partie_I_MLP.ipynb`, `Partie_II_CNN.ipynb`, `Partie_III_Seq2Seq.ipynb`.*
