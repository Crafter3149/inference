"""
inference-aie: Single-wheel distribution for AIE inference server.

Includes inference/, inference_models/, inference_cli/, inference_sdk/,
and the pre-built dashboard (inference/landing/out/).

Torch/torchvision must be installed separately (GPU vs CPU matters).

Install:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install inference_aie-X.Y.Z-py3-none-any.whl
"""

import os
import sys

import setuptools
from setuptools import find_packages

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root)
from inference.core.version import __version__

with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements(path):
    if not isinstance(path, list):
        path = [path]
    requirements = []
    for p in path:
        full = os.path.join(root, p)
        with open(full) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements


def collect_landing_out():
    """Collect all files under inference/landing/out/ as package_data globs.

    landing/out/ is not a Python package, so we express paths relative to the
    nearest parent package (``inference``).  setuptools package_data supports
    recursive globs via ``**``.
    """
    out_dir = os.path.join(root, "inference", "landing", "out")
    if not os.path.isdir(out_dir):
        return []
    patterns = []
    for dirpath, _dirs, files in os.walk(out_dir):
        # skip node_modules or other dev artifacts
        _dirs[:] = [d for d in _dirs if d not in ("node_modules", ".next")]
        rel = os.path.relpath(dirpath, os.path.join(root, "inference"))
        for f in files:
            patterns.append(os.path.join(rel, f).replace(os.sep, "/"))
    return patterns


# inference_models is a sub-project: the actual package lives at
# inference_models/inference_models/.  We discover it separately and
# use package_dir to map it into the wheel.
main_packages = find_packages(
    where=root,
    include=[
        "inference",
        "inference.*",
        "inference_cli",
        "inference_cli.*",
        "inference_sdk",
        "inference_sdk.*",
    ],
    exclude=[
        "tests",
        "tests.*",
    ],
)

im_root = os.path.join(root, "inference_models")
im_packages = find_packages(
    where=im_root,
    include=[
        "inference_models",
        "inference_models.*",
    ],
)

all_packages = main_packages + im_packages

# package_dir: main packages resolve from root, inference_models from its sub-project
package_dir = {"": root}
for pkg in im_packages:
    package_dir[pkg] = os.path.join(im_root, pkg.replace(".", os.sep))

setuptools.setup(
    name="inference-aie",
    version=__version__,
    author="NST",
    description="AIE inference server — single-wheel distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=all_packages,
    package_dir=package_dir,
    include_package_data=False,
    package_data={
        "inference": collect_landing_out(),
        "inference.models.perception_encoder.vision_encoder": [
            "bpe_simple_vocab_16e6.txt.gz"
        ],
        "inference_models": [
            "models/rfdetr/dinov2_configs/*.json",
            "models/perception_encoder/vision_encoder/bpe_simple_vocab_16e6.txt.gz",
        ],
    },
    entry_points={
        "console_scripts": [
            "inference=inference_cli.main:app",
        ],
    },
    install_requires=read_requirements(
        [
            "requirements/_requirements.txt",
            "requirements/requirements.http.txt",
            "requirements/requirements.cli.txt",
            "requirements/requirements.sdk.http.txt",
        ]
    ),
    extras_require={
        "models": [
            # Heavy inference_models deps (transformers, SAM, etc.)
            # Only needed if running non-AIE foundation models
            "transformers>=5.2.0,<5.3.0",
            "timm>=1.0.0,<2.0.0",
            "accelerate>=1.0.0,<2.0.0",
            "einops>=0.7.0,<1.0.0",
            "peft>=0.18.1",
            "segmentation-models-pytorch>=0.5.0,<1.0.0",
            "easyocr~=1.7.2",
            "sentencepiece>=0.2.0,<0.3.0",
            "rf-clip==1.1",
            "rf-segment-anything==1.0",
            "rf-sam-2==1.0.3",
            "rf_groundingdino==0.3.0",
            "python-doctr[torch]>=1.0.0",
            "bitsandbytes>=0.46.1,<0.48.0; sys_platform != 'darwin'",
            "pyvips>=2.2.3,<3.0.0",
            "num2words~=0.5.14",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.13",
)
