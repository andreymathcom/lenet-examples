# Code Examples

Placement: Data → MNIST Demo.

## Approximating a convolutional network with a quadratic polynomial

Can a simple polynomial approximate the outputs of a trained neural network? In this experiment I fit an affine map and a degree-two polynomial to the ten logits of a convolutional teacher on MNIST.

### Mathematical model

For a digit image x, normalize and project onto 100 principal components fitted on the training data:

z = V (vec(x) / 255 − μ).

Construct 5,151 coordinates: 100 PCA scores, 5,050 products zᵢzⱼ with i ≤ j, and a constant. Fit a linear readout to the teacher logits by float64 least squares:

Ŷ = [z, {zᵢzⱼ}, 1] W.

Each output is an affine plus quadratic form in z. The teacher takes raw float pixels in [0,255]; the PCA features use normalized pixels.

### Recorded results

| Model | Training accuracy | MNIST test accuracy |
|---|---:|---:|
| This ONNX convolutional teacher | 97.953% | 97.370% |
| PCA-100 quadratic approximation | 97.058% | 96.470% |

The quadratic figures are saved outputs from my original notebooks; the teacher test accuracy has been independently reproduced. The quadratic approximation loses 0.90 percentage points of test accuracy. Its test agreement with the teacher is 96.370%; test logit RMSE is about 14.39.

### Computation and storage

The product implementation uses 135,744 multiplies and 130,584 adds per image, compared with 13,883,904 of each for this particular teacher. It stores 130,694 floating-point numbers, including μ, V and W, compared with 3,274,634 for the teacher. This amounts to approximately 100 times fewer arithmetic operations and 25 times fewer stored floats. These counts describe the algorithms; actual speed depends on implementation and hardware. This teacher differs from canonical LeNet-5.

### What you can explore

- Fit an affine map to cached teacher logits.
- Fit a PCA-quadratic map and evaluate it on held-out samples.
- Reproduce the full 100-PC experiment with the original ONNX teacher.
- Compare direct polynomial evaluation with diagonalized quadratic forms.
- Inspect the exploratory notebooks, including cases where interpolation fits training data but generalizes poorly.

### Selected code

```python
z = (X / 255.0 - mu) @ V.T
i, j = np.triu_indices(z.shape[1])
phi = np.column_stack((z, z[:, i] * z[:, j], np.ones(len(z))))
W = np.linalg.lstsq(phi, teacher_logits, rcond=None)[0]
```

### Run and inspect

[Source code and original experiments](https://github.com/andreymathcom/lenet-examples)

[Run the small demo in Colab](https://colab.research.google.com/github/andreymathcom/lenet-examples/blob/main/01_cached_logits_demo.ipynb)

[Run the full PCA-100 experiment in Colab](https://colab.research.google.com/github/andreymathcom/lenet-examples/blob/main/02_full_pca100_quadratic.ipynb)

 The small cached-logit demo runs without teacher weights. The full experiment includes the original lenet.onnx and requires sufficient RAM.

This is an approximation of the teacher logits. It does not provide uniform 10⁻⁴ logit accuracy. Diagonalizing each quadratic form changes its representation but increases arithmetic in the implementation shown here.

The supplied teacher model has now been independently verified on all 10,000 test images, reproducing 97.37% accuracy. The quadratic model figures remain recorded results from the original experiment.

