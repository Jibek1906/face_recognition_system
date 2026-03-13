import os
with open("run.py", "r") as f:
    text = f.read()

prefix = """import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
"""

if "OMP_NUM_THREADS" not in text:
    with open("run.py", "w") as f:
        f.write(prefix + text)
