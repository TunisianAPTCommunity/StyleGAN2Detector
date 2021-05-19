import sys
import glob
import os
from PIL import Image

def resize(imgPath):
    size = 224, 224
    try:
        im = Image.open(imgPath)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(imgPath, "JPEG")
    except IOError:
        print("cannot create thumbnail for '%s'" % imgPath)


root_dir = sys.argv[1]
if root_dir.endswith(".jpg") == False:
    root_dir += "/"
files = glob.glob(root_dir + "*.jpg")
i = 0
while i < len(files):
    resize(files[i])
    i += 1