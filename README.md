# CAR-PUF Challenge-Response Prediction

**CS771 - Introduction to Machine Learning | IIT Kanpur**

This project studies how a CAR-PUF with a 32-bit challenge can be represented as a linear classification problem using a **528-dimensional feature map**. The assignment report derives the mapping and compares **LinearSVC** and **Logistic Regression**, examining how the hyperparameters **C** and **tol** affect prediction accuracy and training time.

**Authors:** Aditya Gupta and Praveen Raj.

This overview summarizes the derivation and eight experiment plots in the authors' nine-page assignment report.

## Project at a glance

| Component | Description |
| --- | --- |
| Task | Predict a CAR-PUF response from a binary challenge |
| Input | 32-bit challenge vector |
| Representation | 528 pairwise features constructed from an augmented challenge representation |
| Models compared | LinearSVC and Logistic Regression |
| Parameters studied | C and tol |
| Evaluation views | Accuracy and training time across parameter settings |

## From 32 challenge bits to 528 features

For a challenge $c=(c_1,\ldots,c_{32})\in\lbrace0,1\rbrace^{32}$, first compute the signed suffix products:

$$
x_i=\prod_{j=i}^{32}(1-2c_j),\qquad i=1,\ldots,32.
$$

Append the constant $x_{33}=1$ to obtain $\tilde{x}=(x_1,\ldots,x_{32},1)$.

The feature map contains every distinct pairwise product:

$$
\phi(c)=(x_i x_j)_{1\leq i\lt j\leq33}\in\mathbb{R}^{528}.
$$

There are $\binom{33}{2}=528$ features: **496 products between the first 32 components** and **32 products with the appended constant**. The latter are simply the original transformed components $x_i$. Since $x_i^2=1$, squared terms contribute to the model's bias rather than requiring additional features.

## Why the transformed model is linear

Let the two arbiter-PUF delay models be

$$
\Delta_w=u^\top x+p,\qquad \Delta_r=v^\top x+q.
$$

Define $z=(u-v,p-q)$. For a nonnegative threshold $\tau$, the CAR-PUF response is 0 when $|\Delta_w-\Delta_r|\leq\tau$ and 1 otherwise. Squaring the threshold condition gives the decision function

$$
f(c)=(z^\top\tilde{x})^2-\tau^2
=\sum_{i\lt j}2z_i z_j x_i x_j+\sum_{i=1}^{33}z_i^2-\tau^2
=W^\top\phi(c)+b,
$$

with $W_{ij}=2z_i z_j$ and $b=\sum_{i=1}^{33}z_i^2-\tau^2$.

Thus the decision is linear in the mapped features, with an intercept:

$$
r(c)=\begin{cases}
0,&W^\top\phi(c)+b\leq0,\\
1,&W^\top\phi(c)+b>0.
\end{cases}
$$

The complete derivation is on **pages 2-5** of the report.

## Hyperparameter experiments

The report presents an accuracy plot and a training-time plot for each of the following comparisons:

| Model | Parameter varied | Report page |
| --- | --- | --- |
| LinearSVC | C | 6 |
| Logistic Regression | C | 7 |
| LinearSVC | tol | 8 |
| Logistic Regression | tol | 9 |

The plots show these qualitative trends within the tested ranges:

- Increasing **C** from very small values improves accuracy before the curves largely plateau.
- The LinearSVC plot shows a sharp training-time increase at large **C**, while the Logistic Regression plot shows milder variation.
- Increasing **tol** generally reduces training time, but sufficiently loose settings cause a substantial loss of accuracy in both models.

These experiments illustrate why parameter selection should consider **accuracy and computational cost together**. The plots should be read as trends; the report does not tabulate exact best scores, selected final settings, or dataset split sizes.

## Report notation

This README follows the cumulative-product definition of $x_i$ on **pages 2 and 5**. The final line on page 4 instead writes $x_i=1-2c_i$, which is inconsistent with those definitions.

The response is written as a piecewise rule above to preserve the report's boundary condition: **equality maps to response 0**. Its alternative sign-function expression requires a special convention at zero to produce that same result.

## Repository contents

```text
.
|-- README.md          # Project overview and report summary
`-- submit.py          # Feature mapping and Logistic Regression implementation
```

## Running the implementation

`submit.py` maps each 32-bit challenge to 528 features, trains a scikit-learn `LogisticRegression` model, and prints test accuracy. The supplied script is preserved as provided; it does not run the full parameter sweeps described in the report.

Install NumPy, scikit-learn, and SciPy. Place `train.dat` and `test.dat` in the working directory, with 32 challenge columns followed by the response label in column 33, then run:

```bash
python submit.py
```

The datasets are not included in this repository.
