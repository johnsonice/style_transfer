# style_transfer
all different ways of style transfer


#### style_net borrowed a lot code from https://github.com/lengstrom/fast-style-transfer

### high level api

```
from style_net import inference
import matplotlib.pyplot as plt

options = {'checkpoint': 'style_net/training_model',
         'device': '/gpu:0'
         }

img = 'style_net/examples/content/chicago.jpg'
net = inference.net(options)

result = net.predict(img)
plt.imshow(result)
net.save('test.jpg',result)
```

