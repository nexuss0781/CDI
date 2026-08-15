# Single Google Colab Cell — CDI M1.2 Compiled Training

Run this entire cell once. It mounts Google Drive, creates or updates the canonical CDI checkout, installs dependencies, loads the previous M1.2 checkpoint cumulatively, excludes the previous M1.2 dataset from the new `r2` dataset variant, trains with the validated compiled fixed-shape CDI path, persists all artifacts, and stops after the M1.2 competency report.

```python
from google.colab import drive
from pathlib import Path
import os
import subprocess

# 1. Mount persistent storage.
drive.mount("/content/drive")

# 2. Clone once, then synchronize the Drive checkout with canonical master.
repo = Path("/content/drive/MyDrive/CDI")
if not repo.exists():
    subprocess.run(
        [
            "git", "clone", "--branch", "master", "--single-branch",
            "https://github.com/nexuss0781/CDI", str(repo),
        ],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", "master"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "master"], check=True)
    subprocess.run(["git", "-C", str(repo), "reset", "--hard", "origin/master"], check=True)

# 3. Install the repository dependencies.
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")],
    check=True,
)

# 4. Configure one isolated, cumulative compiled M1.2 round.
os.environ["CDI_DRIVE_ROOT"] = "/content/drive/MyDrive/CDI"
os.environ["CDI_STAGE"] = "m1.2"
os.environ["CDI_PARENT_STAGE"] = "m1.2"
os.environ["CDI_PARENT_DATA_VARIANT"] = "base"
os.environ["CDI_DATA_VARIANT"] = "r2"
os.environ["CDI_NEW_SESSION"] = "1"
os.environ["CDI_SESSION_ID"] = "m1_2_compiled_r2_" + __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
os.environ["CDI_COMPILE"] = "1"
os.environ["CDI_COMPILE_MODE"] = "reduce-overhead"
os.environ["CDI_SKIP_INSTALL"] = "1"

# 5. Run exactly one M1.2 session. The script persists checkpoints, logs,
# datasets, reports, and hashes below /content/drive/MyDrive/CDI/module1/M1.2/.
subprocess.run(["bash", "run.sh"], cwd=repo, check=True, env=os.environ.copy())

print("\nCOMPLETED: compiled CDI M1.2 session")
print("SESSION:", os.environ["CDI_SESSION_ID"])
print("REPORT ROOT:", repo / "module1" / "M1.2" / "sessions" / os.environ["CDI_SESSION_ID"] / "reports")
print("CHECKPOINT ROOT:", repo / "module1" / "M1.2" / "sessions" / os.environ["CDI_SESSION_ID"] / "checkpoints")
```

## Expected behavior

The session uses the previous M1.2 candidate checkpoint as its parent and uses only new documents for the `r2` dataset variant. It keeps the compiled recurrent path fixed at batch size 2 on CPU or batch size 8 on CUDA, with sequence chunks padded to the configured length of 64. The compiled recurrence uses full vocabulary cross-entropy outside the compiled recurrent wrapper, which is appropriate for the current nano configuration.

After completion, copy and return the contents of the generated `M1.2_REPORT.md`. The competency verdict remains separate from the engineering execution result: a finite compiled run with checkpoint persistence does not automatically mean that the M1.2 repetition-control competency has passed.
