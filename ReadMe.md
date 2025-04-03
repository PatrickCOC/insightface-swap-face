Testing
```
python3 -m venv venv 
source venv/bin/activate   # macOS/Linux  
venv\Scripts\activate      # Windows
pip freeze > requirements.txt
```

```
pip install -r requirements.txt
```

```
python3 server.py
```

path: venv/lib/python3.10/site-packages/basicsr/data/degradations.py
```
from torchvision.transforms.functional_tensor import rgb_to_grayscale

to:

from torchvision.transforms.functional import rgb_to_grayscale
```