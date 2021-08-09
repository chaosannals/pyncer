from torch import nn


class TextPyncerNet(nn.Module):
    '''
    文本验证码识别网络。
    '''

    def __init__(self, length, charset_length):
        '''
        初始化。
        '''

        super().__init__()
        self.length = length
        self.charset_length = charset_length

        self.lv1 = nn.Sequential(
            nn.Conv2d(3, 5, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.lv2 = nn.Sequential(
            nn.Conv2d(5, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.lv3 = nn.Sequential(
            nn.Conv2d(10, 16, 6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.lv4 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )


        for i in range(length):
            n = f'lv{5 + i}'
            v = nn.Linear(128, charset_length)
            setattr(self, n, v)
        

    def forward(self, x):
        '''
        前向。
        '''

        x = self.lv1(x)
        x = self.lv2(x)
        x = self.lv3(x)

        x = x.view(-1, 768)
        x = self.lv4(x)

        fn = [getattr(self, f'lv{5 + i}') for i in range(self.length)]

        return [f(x) for f in fn]
