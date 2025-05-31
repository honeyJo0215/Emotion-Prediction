import os, shutil

print(">>> PATH:", os.environ.get("PATH"))
print(">>> ffmpeg 위치:", shutil.which("ffmpeg"))
print(">>> ffplay 위치:", shutil.which("ffplay"))
