# [pyncer](https://github.com/chenshenchao/pyncer)

## 使用

```bash
# 安装
pip install pyncer
```

### 训练

```python
from datetime import datetime
from pyncer.text import TextPyncer, TextPyncerTrainer
from pyncer.text.util import load_by_filename
from tqdm import tqdm

if __name__ == '__main__':
    pyncer = TextPyncer(
        path='./captcha.m',
        length=5,
        charset=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',],
    )

    data = load_by_filename('./debug/')
    batch_size = 128
    trainer = TextPyncerTrainer(
        pyncer=pyncer,
        train_data = data,
        test_data = data,
        batch_size=128
    )

    avg_loss = 0.0
    avg_limit = 100
    avg_count = 0
    for i in range(200):
        for j, loss in tqdm(trainer.train()):
            avg_loss += loss
            c = avg_count + j
            if (c % avg_limit) == 0:
                print(f'count: {c} | loss:{avg_loss / avg_limit}')
                avg_loss = 0.0
        avg_count += j
        accuracy = trainer.test(10)
        print(f'epoch: {i} | accuracy: {accuracy * 100:.2f}%')
        if (i % 3) == 2:
            pyncer.save()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'epoch: {i} | save: {now}')
```

### 猜验证码

```python
import random
from pyncer.text import TextPyncer
from pyncer.text.util import load_by_filename

if __name__ == '__main__':
    pyncer = TextPyncer(
        path='./captcha.m',
        length=5,
        charset=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',],
    )
    data = load_by_filename('./debug/')
    for _ in range(10):
        i = random.randint(0, len(data))
        c, p = data[i]
        r = pyncer.infer(p)
        print(f'{c} <=> {r}')
```
