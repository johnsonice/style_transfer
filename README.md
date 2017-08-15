# style_transfer
all different ways of style transfer

### high level api

```
from style_net import evaluate
import scipy.misc

options = {'checkpoint_dir': 'style_net/training_model',
         'device': '/gpu:0',
         'in_path': 'style_net/examples/content/chicago.jpg'}

sess,in_node,out_node = evaluate.load(options)
results = evaluate.predict(sess,in_node,out_node,options)

scipy.misc.imshow(results)
```

