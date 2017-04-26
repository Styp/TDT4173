import scipy
from PIL import Image
from skimage import filters

from skimage import io, color, exposure, morphology
from skimage import color
import numpy as np

fullPath = '../chars74k-lite/a/a_0.jpg'

rawImg = io.imread(fullPath, as_grey=True)




otsuThreshold = filters.threshold_otsu(rawImg)
img_bw = rawImg > otsuThreshold
intArr = np.array(img_bw).astype(int)
output = np.multiply(intArr,255)

print(output)

io.imsave("test.jpg", output)
