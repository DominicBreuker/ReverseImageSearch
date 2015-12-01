import graphlab as gl
from PIL import Image
from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def save_image(image, id):
    pixels = list(grouper(image.pixel_data, 3))
    img = Image.new("RGB", (256,350))
    img.putdata(pixels)
    img.save("./dresses_resized/image_%s.jpg" % str(id))
    id

sf = gl.SFrame('./data/my_image_data')
sf['image'] = gl.image_analysis.resize(sf['image'], 256, 350, 3)
images = sf['image']

[save_image(image, id) for id, image in enumerate(images)]