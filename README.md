# style_transfer
all different ways of style transfer


#### style_net borrowed a lot code from https://github.com/lengstrom/fast-style-transfer


### download pretrained models 
```
##down load model ckpt files into model folder 
cd style_net/model 
wget https://www.dropbox.com/s/one2ibrtro7tiuv/style_model.tar.gz
tar -xzf style_model.tar.gz
rm style_model.tar.gz
```

#### Some results 
<p align = 'center'>

<img src="result_samples/2.jpg" height=170>
<img src="result_samples/2_tl.jpg" height=170>
<img src="result_samples/2_wave.jpg" height=170>
</p>
<p align = 'center'>

<img src="result_samples/car.jpg" height=170>
<img src="result_samples/car_screem.jpg" height=170>
<img src="result_samples/car_wave.jpg" height=170>
</p>
<p align = 'center'>

<img src="result_samples/5.jpg" height=170>
<img src="result_samples/5_screem.jpg" height=170>
<img src="result_samples/5_wave.jpg" height=170>
</p>
<p align = 'center'>

<img src="result_samples/1.jpg" height=170>
<img src="result_samples/1_totoro.jpg" height=170>
<img src="result_samples/1_wave.jpg" height=170>
</p>

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

