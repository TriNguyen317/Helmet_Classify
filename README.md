# HELMET_CLASSIFY

This project focuses o helmet classify in images and videos using computer vision techniques. Project provides 2 model are trained in our data for users.

## Table of Contents

- [Installation](#installation)
- [Script Documentation](#Script-Documentation)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TriNguyen317/Helmet_Classify.git

   ```

2. Install the required dependencies:

3. Download the model weights and place them in the appropriate directory:

   - You can download the model weights file or use existing weights in the `Model` directory.

## Script documentation

### Model:
   - Project using 2 model: 
      * EfficientNetV2: 
         + Paper: https://arxiv.org/pdf/2104.00298v3.pdf
      * WATT_EffNet: 
         + Paper: https://arxiv.org/pdf/2304.10811.pdf
   - Model and pre_trained of WATT_EffNet are existing in the `Model` and `Best_weight`
   - Model and pre_trained EfficientNetV2 are downloaded by using torchvision library 

### Data:
```
- train
   - NameClass1
      - 1.jpg
      - 2.jpg
   - NameClass2
      - 1.jpg
      - 2.jpg
- valid 
   - NameClass1
      - 1.jpg
      - 2.jpg
   - NameClass2
      - 1.jpg
      - 2.jpg
- test 
   - NameClass1
      - 1.jpg
      - 2.jpg
   - NameClass2
      - 1.jpg
      - 2.jpg

```
- Link data: https://drive.google.com/file/d/1ZgDkSPyIgi0PQ9PSm4O8uyL3DSC5b2Pe/view?usp=sharing
### Args:
    model (str): Path of model. Default: ./Model/WATT-EffNet.pt
    train_dir (str): Path of training data. Default: ./data_classify/train
    val_dir (str): Path of valid data. Default: ./data_classify/val
    test_dir (str): Path of test data. Default: ./data_classify/test
    epoch (int): Num epoch. Default: 50
    batch_size (int): Batch size. Default: 12
    img_size (int): Size of input image. Default:52
    checkpoint (int): Path of checkpoint/pre_trained to load in the model. 
    lr (float): Learning rate. Default: 0.0001
    momentum (float): Momentum for optimizer. Default: 0.8
    train (bool): Task is training. Default: False
    test (bool): Task is testing. Default: False

### Train with no pre-trained
```bash
python main.py --train --model ./Model/EfficientNetV2.pt --momentum 0.5 --batch_size 24 --epoch 100 --lr 0.00001
```

### Train with pre_trained/checkpoint
```bash
python main.py --train --model ./Model/EfficientNetV2.pt --momentum 0.5 --batch_size 24 --epoch 100 --lr 0.00001 --checkpoint ./Checkpoint/EfficientNetV2.pt
```

### Testing
```bash
python main.py --test --batch_size 12 --model ./Model/WATT_EffNet.pt --checkpoint ./Best_weight/WATT_EffNET2.pt
```

### Classify
```bash
python main.py --classify  --model ./Model/WATT_EffNet.pt --checkpoint ./Best_weight/WATT_EffNET2.pt --image 1.jpg
```

## Contact

For any inquiries or questions, please contact:

- Project Maintainer: Nguyen Dinh Tri (dinhtrikt11102002@gmail.com)
- Project Homepage: [https://github.com/TriNguyen317/Ship-detection-and-tracking]

Feel free to reach out with any feedback or suggestions!