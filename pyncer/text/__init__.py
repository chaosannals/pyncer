import os
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchnet import meter
from PIL import Image
from .data import TextPyncerDatum, TextPyncerDataset
from .model import TextPyncerNet

class TextPyncer:
    '''
    文本验证码识别器。
    '''

    def __init__(self, path, length, charset, size=(128, 64)):
        '''
        '''

        self.path = path
        self.datum = TextPyncerDatum(length, charset, size)
        self.net = TextPyncerNet(length, len(charset))
        if os.path.isfile(path):
            self.load()
    
    def load(self):
        '''
        加载数据。
        '''

        d = torch.load(self.path)
        self.net.load_state_dict(d)

    def save(self):
        '''
        保存数据。
        '''

        d = self.net.state_dict()
        torch.save(d, self.path)


    def infer(self, path):
        '''
        '''
        
        i = Image.open(path).convert('RGB')
        w = self.datum.size[0]
        h = self.datum.size[1]
        v = self.datum.transform(i).view(1, 3, w, h)
        r = [y.topk(1, dim=1)[1] for y in self.net(v)]
        c = torch.cat(r, dim=1)
        return self.datum.cast_text(c[0])


class TextPyncerTrainer:
    '''
    文本验证码识别器训练器。
    '''

    def __init__(self, pyncer: TextPyncer, train_data, test_data, loader_worker_count=4, batch_size=128, learning_rate=1e-3):
        '''
        '''

        self.pyncer = pyncer
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.pyncer.net.parameters(), lr=learning_rate)
        self.loss_meter = meter.AverageValueMeter()
        train_dataset = TextPyncerDataset(pyncer.datum, train_data)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=loader_worker_count)
        test_dataset = TextPyncerDataset(pyncer.datum, test_data)
        self.test_laoder = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=loader_worker_count)

    def train(self):
        '''
        训练。
        '''

        for i, (x, code) in enumerate(self.train_loader, 0):
            self.optimizer.zero_grad()
            r = self.pyncer.net(x)
            l = self.pyncer.datum.length
            loss = sum([self.criterion(r[k], code[:,k]) for k in range(l)])
            loss_value = loss.item()
            self.loss_meter.add(loss_value)
            loss.backward()
            self.optimizer.step()
            yield i + 1, loss_value

    def test(self, test_count):
        '''
        测试。
        '''

        total_count = test_count * self.batch_size
        right_count = 0
        for i, (x, code) in enumerate(self.test_laoder):
            if i >= test_count:
                break
            r = [y.topk(1, dim=1)[1].view(self.batch_size, 1) for y in self.pyncer.net(x)]
            y = torch.cat(r, dim=1)
            diff = (y != code)
            diff = diff.sum(1)
            diff = (diff != 0)
            result = diff.sum(0).item()
            right_count += (self.batch_size - result)
        return float(right_count) / float(total_count)



