from matplotlib import transforms

class LabelToBinary(object):
    def __call__(self, sample):
        if sample == 'malignant' or sample == 1.0:
            return 1
        else:
            return 0
        
class Rotate(object):
    def __init__(self, angle = 270):
        self.angle = angle
        
    def __call__(self, sample):
        return transforms.functional.rotate(sample, self.angle)
    
class HorizontalFlip(object):
    def __call__(self, sample):
        return transforms.functional.hflip(sample)
    
class VerticalFlip(object):
    def __call__(self, sample):
        return transforms.functional.vflip(sample)