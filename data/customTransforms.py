from matplotlib import transforms

# Convert ISIC dataset to binary labels
class IsicToBinary(object):
    def __call__(self, sample):
        if sample == 'malignant' or sample == 1.0:
            return 1
        else:
            return 0

# Convert HAM dataset to binary labels   
class HamToBinary(object):
    def __call__(self, sample):
        if sample == 'bkl':
            return 1
        else:
            return 0

# Rotate the image
class Rotate(object):
    def __init__(self, angle = 270):
        self.angle = angle
        
    def __call__(self, sample):
        return transforms.functional.rotate(sample, self.angle)

# Flip the image horizontally
class HorizontalFlip(object):
    def __call__(self, sample):
        return transforms.functional.hflip(sample)

# Flip the image vertically
class VerticalFlip(object):
    def __call__(self, sample):
        return transforms.functional.vflip(sample)