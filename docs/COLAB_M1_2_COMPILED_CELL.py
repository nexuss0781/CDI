from google.colab import drive
from pathlib import Path
import os
import subprocess
from datetime import datetime, timezone

drive.mount("/content/drive")

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

subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")],
    check=True,
)

os.environ["CDI_DRIVE_ROOT"] = "/content/drive/MyDrive/CDI"
os.environ["CDI_STAGE"] = "m1.2"
os.environ["CDI_PARENT_STAGE"] = "M1.2"
os.environ["CDI_PARENT_DATA_VARIANT"] = "base"
os.environ["CDI_DATA_VARIANT"] = "r2"
os.environ["CDI_NEW_SESSION"] = "1"
os.environ["CDI_SESSION_ID"] = "m1_2_compiled_r2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
os.environ["CDI_COMPILE"] = "1"
os.environ["CDI_COMPILE_MODE"] = "reduce-overhead"
os.environ["CDI_SKIP_INSTALL"] = "1"

subprocess.run(["bash", "run.sh"], cwd=repo, check=True, env=os.environ.copy())

print("\nCOMPLETED: compiled CDI M1.2 session")
print("SESSION:", os.environ["CDI_SESSION_ID"])
print("REPORT ROOT:", repo / "module1" / "M1.2" / "sessions" / os.environ["CDI_SESSION_ID"] / "reports")
print("CHECKPOINT ROOT:", repo / "module1" / "M1.2" / "sessions" / os.environ["CDI_SESSION_ID"] / "checkpoints")
