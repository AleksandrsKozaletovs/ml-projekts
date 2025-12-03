from PIL import Image
import glob

path = 'data/images/'

cars = glob.glob(path + '*.png')
for c in cars:
    file_name = c.split('.')[0]
    im = Image.open(c)
    im = im.convert('RGB')
    im.save(f'{file_name}.jpeg', quality=95)
    