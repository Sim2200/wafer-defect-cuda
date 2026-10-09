"""Profiling-only TensorRT run (Nsight Systems + Nsight Compute), pushed by
scripts/run_on_kaggle.py --experiment nsight. Runs kaggle/run_trt.py with NSIGHT_ONLY=1: train, export,
build the FP16 engine, profile; the benchmark and accuracy files are left untouched."""

import glob
import os
import runpy

os.environ["NSIGHT_ONLY"] = "1"
runpy.run_path(glob.glob("/kaggle/input/**/kaggle/run_trt.py", recursive=True)[0], run_name="__main__")
