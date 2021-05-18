import sys
import requests
import os

outputDir = sys.argv[1]
if outputDir.endswith("/") == False:
    outputDir += "/"
try:
    os.mkdir(outputDir)
except Exception:
    pass
i = 0
while i < 1000:
    r = requests.get("https://thispersondoesnotexist.com/image")
    with open(outputDir + str(i) + ".jpg", "wb") as imgFile:
        imgFile.write(r.content)
    i += 1