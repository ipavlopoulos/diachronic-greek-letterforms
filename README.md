# Representation Learning of Ancient Greek Letterforms across Time

`TL;DR` Domain-aware contrastive learning yields robust embeddings for evolving, low-resource handwriting data.

You may find the data and the code, along with a saved model (a lightweight CNN only) for fast inference. This repository is anonymised.

# The Algorithm

The image classification method uses **Similarity-Weighted Supervised Contrastive Loss (SW-SCL)** for 2D image classification with CNNs, and is enhanced with a **novel lacunae-inspired augmentation strategy**. 
Optionally, one may use **Test-Time Augmentation (TTA)** for embeddings and **expert-defined class similarity priors**.

---

## Problem Setup

Let $D = \{(x_i, y_i)\}_{i=1}^N$ be a labeled dataset, with $x_i \in \mathbb{R}^{C \times H \times W}$ and $y_i \in \{1, \dots, C\}$.  

We aim to learn a **feature embedding function** $f_\theta: x \mapsto z \in \mathbb{R}^D$ such that embeddings of samples from the same class are close, and embeddings of samples from different classes are separated according to **class similarity**.

---

## Supervised Contrastive Loss

The standard **Supervised Contrastive Loss ** for an anchor $i$ is:

$$
L_i^{sup} = - \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}
$$

where:

- $P(i) = \{ j \neq i \mid y_j = y_i \}$ is the set of positives for anchor $i$,  
- $z_i$ are L2-normalized embeddings,  
- $\tau > 0$ is the temperature hyperparameter.  

---

## Similarity-Weighted Negatives

We extend SupCon by **weighting negatives** according to **class similarity**. Let $S \in [0,1]^{C \times C}$ be a precomputed class similarity matrix with zero diagonal. Then for negative pair $(i,a)$:

$$
w_{ia} = 1 + \lambda \frac{S_{y_i, y_a}}{\bar{S}}, \quad \text{for } y_i \neq y_a
$$

where:

- $\lambda > 0$ is a weighting factor,  
- $\bar{S}$ is the mean off-diagonal similarity,  
- $w_{ia} = 1$ for positives or diagonal entries.  

The **Similarity-Weighted loss** is then:

$$
L_i^{SW-SCL} = - \frac{1}{|P(i)|} \sum_{p \in P(i)} 
\log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} w_{ia}  \exp(z_i \cdot z_a / \tau)}
$$

The final batch loss is:

$$
L^{SW-SCL} = \frac{1}{B} \sum_{i=1}^{B} L_i^{SW-SCL}
$$

---

## Expert Priors (optional)

If **expert knowledge** about visual similarity of classes is available, we can define a prior matrix $S_{prior} \in [0,1]^{C \times C}$. The **final similarity matrix** used in SW-SCL is a blend:

$$
S_{final} = (1 - \alpha_{prior}) S_{dynamic} + \alpha_{prior} S_{prior}
$$

- $\alpha_{prior} \in [0,1]$ controls the contribution of the expert prior.  
- Letters in the same visual group (e.g., straight lines, curves, triangles) have high similarity, others have low similarity.

---

## Total Loss for Classification

We combine the SW-SCL with standard **cross-entropy (CE) loss**:

$$
L = L_{CE} + \lambda_{SCL}  L^{SW-SCL}
$$

- $\lambda_{SCL}$ controls the contribution of the contrastive term.  

---

## Test-Time Augmentation (TTA)

To improve embedding quality, we optionally apply **TTA**. For each sample $x_i$, we generate $n$ augmented views $\{x_i^{(1)}, \dots, x_i^{(n)}\}$ and compute embeddings:

$$
z_i^{(k)} = f_\theta(x_i^{(k)}), \quad k=1,\dots,n
$$

The SW-SCL is then computed over all augmented embeddings, effectively increasing positive samples and improving robustness.

---

## Similarity Matrix Construction

We construct $S_{dynamic}$ dynamically from **class prototypes**:

1. Compute embeddings for all training samples: $z_i = f_\theta(x_i)$.  
2. Compute class prototypes:  

$$
\mu_c = \frac{1}{|C_c|} \sum_{i: y_i = c} z_i, \quad \mu_c = \frac{\mu_c}{\| \mu_c \|_2}
$$

3. Compute pairwise cosine similarities and clamp to $[0,1]$:

$$
S_{c_1,c_2} = \max(0, \min(1, \mu_{c_1} \cdot \mu_{c_2})), \quad S_{c,c}=0
$$

Optionally, $S$ is **updated every few epochs** or smoothed with an **exponential moving average** for stability.

---
