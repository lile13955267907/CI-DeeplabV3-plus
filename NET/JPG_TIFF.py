import glob       #jpg转tif
from PIL import Image
for i in glob.glob(r'2.jpg'):
    im = Image.open(i,"r")
    print(i.split(".")[0])
    im.save("{}_new.tif".format(i.split(".")[0]),quality=95)