import torch
from torch.nn import Conv2d
from torch import nn 
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from GlobalPooling import *
from Spatial_Attention import Spatial_Attention
from Channel_Attention import Channel_Attention
from MBConvBlock import MBConvBlock
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import cv2

# nvidia-smi -l
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WATT_EffiNet model
class WATT_EffiNet(torch.nn.Module):
    def __init__(self, expansion = 6) -> None:
        super(WATT_EffiNet, self).__init__()
        self.Layer_1_conv = torch.nn.Sequential(
            Conv2d(3,32,(3,3),stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.Layer_2_MBConv = MBConvBlock(32,16*expansion)
        
        self.Layer_3_MBConv = MBConvBlock(16*expansion,24*expansion)
        
        self.Layer_4_MBConv = MBConvBlock(24*expansion,24*expansion)
        
        self.Layer_5_MBConv = MBConvBlock(24*expansion,40*expansion)
        
        self.Layer_6_MBConv = MBConvBlock(40*expansion,80*expansion)
        
        self.Layer_7_MBConv = MBConvBlock(80*expansion,112*expansion)
        
        
        self.Channel_Attention = Channel_Attention(112*expansion)
        self.Spatial_Attention = Spatial_Attention()
        
        
        self.GAvgpool_1 = GlobalAvgPooling()
        self.fc_1 = nn.Linear(in_features=2704,out_features=512)
        self.dropout_1 = nn.Dropout(0.3)
        self.ReLU_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=512,out_features=2)
        self.dropout_2 = nn.Dropout(0.2)
        self.ReLU_2 = nn.ReLU()
        #self.sortmax = nn.Softmax(dim=0)

    def forward(self, input):
        x = self.Layer_1_conv(input)
        x = self.Layer_2_MBConv(x)
        x = self.Layer_3_MBConv(x)
        x = self.Layer_4_MBConv(x)
        x = self.Layer_5_MBConv(x)
        x = self.Layer_6_MBConv(x)

        
        input_cbamx = self.Layer_7_MBConv(x)
        cbam_feature = self.Channel_Attention(input_cbamx)
        cbam_feature = self.Spatial_Attention(cbam_feature)
        
        output = input_cbamx + cbam_feature
        
        output = self.GAvgpool_1(output,1)
        output = self.dropout_1(output)
        output = torch.flatten(output, start_dim=1)
        output = self.fc_1(output)
        #output = self.ReLU_1(output)
        output = self.fc_2(output)
        #output = self.ReLU_2(output)
                 
        return output

# Load data with batch-size
class CustomDataset(object):
    def __init__(self, train_dir, val_dir, test_dir, img_size = (224,224), n_classes = 2):
        # ''' default_transform = transforms.Compose([
        #     transforms.Resize((224,224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        # '''
        train_transform = transforms.Compose([
        # Zoom(5,(224,224)),
        transforms.RandomRotation((0,15)),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(0.2),
        transforms.Resize(img_size),
        transforms.ToTensor()])

        val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()])
        
        self.train_data = datasets.ImageFolder(train_dir, transform=train_transform)
        self.val_data = datasets.ImageFolder(val_dir, transform=val_transform)
        self.test_data = datasets.ImageFolder(test_dir, transform=val_transform)

    def load(self, batch_size, num_workers = 1):
        train_loader = DataLoader(self.train_data,batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.val_data,batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(self.test_data,batch_size=batch_size, shuffle=True, num_workers=num_workers)
        print('Data loaded!')
        return train_loader, val_loader, test_loader

# ___Training function___
def Training_model(args, model, train_data, valid_data):
    # Read data
    num_train = len(train_data.dataset.targets)
    num_val = len(valid_data.dataset.targets)
    num_epochs = args.epochs
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    best_accuracy = 0.9
    a_epochs = []
    a_accuracy =[]
    
    # Start training 
    for epoch in range(num_epochs):
        print("-----Epoch %d-----:" %(epoch+1)) 
        running_loss = 0.0
        model.train()
        # num_batch = 0
        with tqdm(total=num_train, ncols=45) as pbar:
            for x_batch, y_batch in train_data:
                
                x_batch = x_batch.to(device) 
                y_batch = y_batch.to(device)
                # optimizer.zero_grad()
                
                pred = model(x_batch).to(device)
                
                loss = loss_func(pred, y_batch)
                # loss.requires_grad = True
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                pbar.update(min(args.batch_size,num_train-pbar.n))
            Avg_loss = running_loss/num_train
            # print('Epoch [%d] loss: %.3f' %(epoch + 1, Avg_loss))
        
        with tqdm(total=num_val, ncols=100) as pbar:   
            running_loss = 0.0
            model.eval()
            num_right_label = 0
            for x_batch, y_batch in valid_data:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                output = nn.Softmax(dim=1)(pred)
                label = []
                for result in output:
                    _, classes = torch.max(result,dim=0)
                    label.append(classes) 
                label = torch.tensor(label)
                for i in range(len(label)):
                    if label[i] == y_batch[i]:
                        num_right_label += 1
                pbar.update(min(args.batch_size,num_val-pbar.n))
            accuracy = num_right_label/num_val
            pbar.set_postfix(Average_Loss=Avg_loss, Accuracy=accuracy)
            a_epochs.append(epoch)
            a_accuracy.append(accuracy)
            
            # Save plot 
            plt.plot(a_epochs, a_accuracy)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.savefig('./Result/Result.png')
            #print("Accuracy: %f" %(accuracy))
            torch.save(model, "./Checkpoint/last.pt")
            if accuracy > best_accuracy:
              torch.save(model,"./Checkpoint/best.pt")
              best_accuracy = accuracy
              print("----Best state----")
    print('Finished Training')

# ___ Testing model ___
def Test_model(args, model, test_data):
    num_test = len(test_data.dataset.targets)
    num_right_label = 0
    
    for x_batch, y_batch in test_data:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        output = nn.Softmax(dim=1)(pred)
        label = []
        for result in output:
            _, classes = torch.max(result,dim=0)
            label.append(classes) 
        label = torch.tensor(label)
        for i in range(len(label)):
            if label[i] == y_batch[i]:
                num_right_label += 1
    accuracy = num_right_label/num_test
    print("Accuracy: ", accuracy)

# ___ Testing Image ___
def Test_image(args, model, test_img):
    pred = model(test_img.to(device))
    output = nn.Softmax(dim=1)(pred)[0]
    _, classes = torch.max(output,dim=0)
    res = "Helmet" if classes==1 else "No-helmet" 
    # img = test_img.reshape(52,52,3).cpu().numpy()
    # cv2.putText(img,res,(5,5),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite("Predict.jpg",img)
    print("Result: ", res)

if __name__ == "__main__":
    """
    Args:
        model (str): Path of model. Default: ./Model/WATT-EffNet.pt
        train_dir (str): Path of training data. Default: ./data_classify/train
        val_dir (str): Path of valid data. Default: ./data_classify/val
        test_dir (str): Path of test data. Default: ./data_classify/test
        epoch (int): Num epoch. Default: 50
        batch_size (int): Batch size. Default: 12
        img_size (int): Size of input image. Default:52
        checkpoint (int): Path of checkpoint to load in the model. 
        lr (float): Learning rate. Default: 0.0001
        momentum (float): Momentum for optimizer. Default: 0.8
        train (bool): Task is training. Default: False
        test (bool): Task is testing. Default: False
        classify (bool): Task is testing. Default: False
        image (str): Path of image. Default: 1.jpg
    """
    parser = argparse.ArgumentParser(prog="Helmet_classify", 
                                     epilog='Text at the bottom of help')
    parser.add_argument("--model", type=str, 
                        default="./Model/EfficientNetV2.pt", help="Path of model")
    parser.add_argument("--train_dir", type=str,
                        default="./data_classify/train", help="Path of training data")
    parser.add_argument("--val_dir", type=str,
                        default="./data_classify/valid", help="Path of valid data")
    parser.add_argument("--test_dir", type=str,
                        default="./data_classify/test", help="Path of test data")
    parser.add_argument("--epochs", type=int,
                        default=50, help="Num epochs")
    parser.add_argument("--batch_size", type=int,
                        default=12, help="Batch size")
    parser.add_argument("--img_size", type=int,
                        default=52, help="Size of input image")
    parser.add_argument("--checkpoint", type=str,
                        default="", help="Path of checkpoint to load in the model")
    parser.add_argument("--lr", type=float,
                        default=0.00001, help="Learning rate for optimizer")
    parser.add_argument("--momentum", type=float,
                        default=0.8, help="Momentum for optimizer")
    parser.add_argument("--train", type=bool, action=argparse.BooleanOptionalAction,
                    default=False, help="Training with a checkpoint file")
    parser.add_argument("--test", type=bool, action=argparse.BooleanOptionalAction,
                default=False, help="Test model by test_dir")
    parser.add_argument("--classify", type=bool, action=argparse.BooleanOptionalAction,
                default=False, help="Classify image")
    parser.add_argument("--image", type=str,
                default="1.jpg", help="Task is classify image")

    args = parser.parse_args()
    
    #Load model
    model = torch.load(args.model)
    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint).state_dict(),strict=True)
    
    train_dir = args.train_dir
    val_dir = args.val_dir
    test_dir = args.test_dir
    batchsize = args.batch_size
    img_sz = args.img_size
    data = CustomDataset(train_dir, val_dir, test_dir, (img_sz,img_sz))
    
    train_data, valid_data, test_data = data.load(batchsize)
        
    model.to(device)
    State = "-----Training-----" if args.train else "-----Testing-----"
    
    print("State: ", State)
    if args.train:
        Training_model(args, model, train_data, valid_data)
    elif args.classify:
        image = Image.open(args.image)
        resize = (args.img_size, args.img_size)
        resized_image = image.resize(resize)
        image = torch.tensor(np.array(resized_image),dtype=torch.float).reshape((1,3,resize[0], resize[1]))
        Test_image(args,model,image)
    elif args.test:
        Test_model(args, model, test_data)
        
    