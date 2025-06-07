import shutil
import os

# Move training images into images/train
shutil.move(r"C:\Users\Sonali Maharana\Desktop\dataset\train", 
            r"C:\Users\Sonali Maharana\Desktop\dataset\images\train")

# Move test images into images/val
shutil.move(r"C:\Users\Sonali Maharana\Desktop\dataset\test", 
            r"C:\Users\Sonali Maharana\Desktop\dataset\images\val")
