
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

class TextPyncerDatum:
    '''
    '''

    def __init__(self, length, charset, size=(128, 64)):
        '''
        '''

        self.length = length
        self.charset = charset
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def cast_code(self, text):
        '''
        '''
        return [self.charset.index(i) for i in text]

    def cast_text(self, code):
        '''
        '''
        return ''.join([self.charset[i] for i in code])

class TextPyncerDataset(data.Dataset):
    '''
    '''

    def __init__(self, datum: TextPyncerDatum, infos):
        '''
        '''

        self.datum = datum
        self.infos = infos

    def __getitem__(self, index):
        '''
        '''

        t, p = self.infos[index]
        a = self.datum.cast_code(t)
        i = Image.open(p).convert('RGB')
        d = self.datum.transform(i)
        return d, torch.Tensor(a).long()

    def __len__(self):
        '''
        '''

        return len(self.infos)

