import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import warnings
warnings.filterwarnings('ignore')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from webcam import run_webcam
from videofile import run_video

print("\n" + "="*60)
print(" Media pipe input mode")
print("="*60)
print("\nSelect video source:")
print("  [1] Webcam")
print("  [2] Video file")
choice = input("\n Enter 1 or 2: ").strip()

if choice == "1":
    run_webcam()

elif choice == "2":
    run_video()

else:
    print("Invalid choice. Run again and enter 1 or 2.")
