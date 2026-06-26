"""
================================================================
Sharing GPU arrays across processes in a scikit-learn GridSearch
================================================================

This example shows joblib's ``share_gpu_arrays`` feature in action through
scikit-learn's :class:`~sklearn.model_selection.GridSearchCV` together with an
estimator that supports the Array API (:class:`~sklearn.linear_model.Ridge`),
operating on a PyTorch CUDA tensor.

``GridSearchCV`` fans out its cross-validation fits over worker processes using
joblib's ``loky`` backend. When the feature matrix ``X`` lives on the GPU,
joblib shares that single device allocation with every worker via CUDA
inter-process communication (IPC) instead of copying it through host memory and
re-uploading it in each worker. This is the GPU counterpart of joblib's numpy
memory-mapping.

It happens **automatically**: ``share_gpu_arrays`` defaults to ``"auto"``, which
shares device arrays larger than ``max_nbytes`` whenever a spawn-based process
backend is used (loky is). No special configuration is required.

Requirements
------------
- a CUDA-capable GPU and PyTorch built with CUDA,
- scikit-learn >= 1.5,
- the environment variable ``SCIPY_ARRAY_API=1`` set *before* scikit-learn (and
  SciPy) are imported, which is what enables Array API dispatch end to end.
"""

# ``SCIPY_ARRAY_API`` must be set before scikit-learn / SciPy are imported.
# Worker processes inherit this environment, so it is active there too.
import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import torch  # noqa: E402

from sklearn import set_config  # noqa: E402
from sklearn.datasets import make_regression  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

import joblib  # noqa: E402


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")

    # Enable scikit-learn's Array API dispatch so estimators compute on the GPU
    # array's namespace (here PyTorch on CUDA). scikit-learn automatically
    # propagates this global configuration to the joblib worker processes.
    set_config(array_api_dispatch=True)

    # Build a regression problem and move the feature matrix to the GPU. We make
    # X comfortably larger than joblib's ``max_nbytes`` (1 MB by default) so that
    # GPU sharing kicks in: 50_000 x 100 float32 = 20 MB.
    X_np, y_np = make_regression(
        n_samples=50_000, n_features=100, noise=0.1, random_state=0
    )
    X = torch.asarray(X_np, dtype=torch.float32, device="cuda:0")
    y = torch.asarray(y_np, dtype=torch.float32, device="cuda:0")
    x_mb = X.element_size() * X.nelement() / 1e6
    print(f"X: {tuple(X.shape)} float32 = {x_mb:.0f} MB on {X.device}")

    search = GridSearchCV(
        Ridge(solver="svd"),
        param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        cv=5,
        n_jobs=4,
    )

    # No joblib configuration is needed: with the default
    # ``share_gpu_arrays="auto"``, the 20 MB GPU ``X`` above is shared with the
    # worker processes via CUDA IPC automatically.
    #
    # To *force* sharing and get a clear error if it is not possible (e.g. a
    # fork-based start method), wrap the fit explicitly:
    #
    #     with joblib.parallel_config(share_gpu_arrays="on"):
    #         search.fit(X, y)
    #
    # and use ``share_gpu_arrays="off"`` to disable it entirely.
    search.fit(X, y)

    print("best params:", search.best_params_)
    print(f"best CV score (R^2): {search.best_score_:.5f}")
    print("best estimator coefficients on GPU:", search.best_estimator_.coef_.is_cuda)

    # Release the shared CUDA tensors held by the (reused) loky workers before
    # the process exits, to avoid a noisy PyTorch CUDA-IPC teardown warning.
    joblib.externals.loky.get_reusable_executor().shutdown(wait=True)


if __name__ == "__main__":
    main()
