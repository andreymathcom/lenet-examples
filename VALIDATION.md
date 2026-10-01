# Validation

The cached-logit notebook was executed end to end on 2026-10-01 with NumPy, 2,000 fitting samples, 1,000 held-out training samples and 20 PCA components (seed 42). Accuracy on that held-out subset: cached teacher 97.8%, affine 75.9%, quadratic 91.2%. These are demonstration results, not the official test-set results of the full experiment. Both prepared notebooks passed Python syntax checks. The full experiment was not run: the original ONNX model was absent and PyTorch/ONNX were not installed in this workspace.

The original ONNX teacher was subsequently supplied, validated, and executed with ONNX Runtime on all 10,000 official test images: 97.37% accuracy. First 100 cached training rows: maximum absolute logit discrepancy 5.340576171875e-05. Verified test logits and the original ONNX model are included. The full PCA-100 fit has not been rerun.
