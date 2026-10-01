# MNIST: Approximating LeNet with a Quadratic Model

By Andrey Melnikov · Code examples for andreymath.com

[Run the small demo in Google Colab](https://colab.research.google.com/github/andreymathcom/lenet-examples/blob/main/01_cached_logits_demo.ipynb)

[Run the full PCA-100 experiment in Google Colab](https://colab.research.google.com/github/andreymathcom/lenet-examples/blob/main/02_full_pca100_quadratic.ipynb)

## Start here

Open the small demo in Colab and run all cells. It downloads this repository and extracts the bundled MNIST data automatically. It fits cached teacher logits using 2,000 fitting images, evaluates on 1,000 disjoint training images, and compares affine and PCA-20 quadratic maps. It needs only NumPy. This is a quick illustration, not the full official test experiment.

For the full experiment, use the second notebook. It includes the original ONNX teacher, installs dependencies, fits 100 principal components and a degree-two readout using all 60,000 training images, then evaluates on the 10,000 official test images. Use a high-memory runtime: the float64 60,000 × 5,151 design matrix alone needs about 2.47 GB; intermediates and least-squares workspace need more memory.

Locally: clone this repository, install requirements.txt, and open a notebook using Jupyter. Setup cells unpack lenet-data.zip and place the teacher in models/ automatically.

## Model

Normalize pixels for PCA: z = V (x/255 − μ). Construct 100 scores, 5,050 upper-triangular products zᵢzⱼ, and a constant: 5,151 coordinates. Fit the linear readout to ten teacher logits using float64 least squares. The teacher itself expects raw float pixels in [0,255].

## Recorded full-experiment results

| Model | Training accuracy | Official MNIST test accuracy |
|---|---:|---:|
| This ONNX teacher | 97.953% | 97.370% |
| PCA-100 quadratic surrogate | 97.058% | 96.470% |

The teacher's 97.37% test accuracy has been independently reproduced using ONNX Runtime on all 10,000 test images. Quadratic-model results are recorded outputs from the supplied original experiment; that full fit has not been rerun here. Test argmax agreement with the teacher was 96.37%; test logit RMSE was about 14.39. This is not a uniformly accurate 10⁻⁴ approximation.

The product-form surrogate uses 135,744 multiplies and 130,584 adds per image, compared with 13,883,904 of each for this teacher. It stores 130,694 floats (mean, PCA basis, readout) versus 3,274,634 for the teacher: roughly 100× fewer arithmetic operations and 25× fewer stored floats. These are theoretical counts, not measured speedups. This particular teacher differs from canonical LeNet-5. Diagonalization increases arithmetic in the implementation shown.

## Files

- 01_cached_logits_demo.ipynb: quick runnable demonstration.
- 02_full_pca100_quadratic.ipynb: portable full experiment with equations, inference functions and operation counts.
- approximation.py: portable NumPy helpers.
- lenet.onnx: original teacher model.
- lenet-data.zip: supplied MNIST IDX data, cached training logits, independently verified test logits.
- original-experiments.zip: original saved experiment notebooks with private local paths redacted. Exploratory very large fits and failed interpolation attempts are retained as research history, not the default demo.
- VALIDATION.md and teacher_validation.json: checks performed.

The cached training logits follow standard MNIST training order, as indicated by the original notebooks and the first-100-row verification. No new license has been imposed; the uploaded project contained no license.
