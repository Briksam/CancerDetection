import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import copy
import os
import torch
from PIL import Image
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import utils






labels_df = pd.read_csv("C:/Users/break/Desktop/project1/train.csv")



labels_df.shape


labels_df[labels_df.duplicated(keep=False)]


labels_df['label'].value_counts()



imgpath = 'C:/Users/break/Desktop/project1/train' 
malignant = labels_df.loc[labels_df['label']==1]['img_id'].values   
normal = labels_df.loc[labels_df['label']==0]['img_id'].values       

print('normal ids')
print(normal[0:3],'\n')

print('malignant ids')
print(malignant[0:3])


torch.manual_seed(0) 

class pytorch_data(Dataset):
    
    def __init__(self,data_dir,transform,data_type="train"):      
    
        
        cdm_data=os.path.join(data_dir,data_type) 
        
        file_names = os.listdir(cdm_data)  
       
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names]   
        
        
        labels_data=os.path.join(data_dir,"train.csv") 
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("img_id", inplace=True) 
        self.labels = [labels_df.loc[filename[:-5]].values[0] for filename in file_names]  
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) 
      
    def __getitem__(self, idx):
       
        image = Image.open(self.full_filenames[idx])  
        image = self.transform(image) 
        return image, self.labels[idx]


import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((195,195))])


data_dir = 'C:/Users/break/Desktop/project1/'
img_dataset = pytorch_data(data_dir, data_transformer, "train") 


img,label=img_dataset[10]
print(img.shape,torch.min(img),torch.max(img))


len_img=len(img_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train


train_ts,val_ts=random_split(img_dataset,
                             [len_train,len_val]) 

print("train dataset size:", len(train_ts))
print("validation dataset size:", len(val_ts))


ii=-1
for x,y in train_ts:
    print(x.shape,y)
    ii+=1
    if(ii>5):
        break



tr_transf = transforms.Compose([
#     transforms.Resize((40,40)),
#    transforms.RandomHorizontalFlip(p=0.5), 
#    transforms.RandomVerticalFlip(p=0.5),  
#    transforms.RandomRotation(45),         
#     transforms.RandomResizedCrop(50,scale=(0.8,1.0),ratio=(1.0,1.0)),
    transforms.ToTensor()])


val_transf = transforms.Compose([
    transforms.ToTensor()])


train_ts.transform=tr_transf
val_ts.transform=val_transf


train_ts.transform


from torch.utils.data import DataLoader


train_dl = DataLoader(train_ts,
                      batch_size=32, 
                      shuffle=True)


val_dl = DataLoader(val_ts,
                    batch_size=32,
                    shuffle=False)


for x,y in train_dl:
    print(x.shape,y)
    break


def findConv2dOutShape(hin,win,conv,pool=2):
    
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)

import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    
    
    def __init__(self, params):
        
        super(Network, self).__init__()
    
        Cin,Hin,Win=params["shape_in"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]  
        num_classes=params["num_classes"] 
        #self.dropout_rate=params["dropout_rate"] 
        
        
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = F.relu(self.fc1(X))
        #X=F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)


params_model={
        "shape_in": (3,195,195), 
        "initial_filters": 8,    
        "num_fc1": 100,
        #"dropout_rate": 0.55,  #0.25
        "num_classes": 2}


cnn_model = Network(params_model)

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
model = cnn_model.to(device)




loss_func = nn.NLLLoss()  #reduction="sum"



from torch import optim
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)  
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)




def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target) 
    pred = output.argmax(dim=1, keepdim=True) 
    metric_b=pred.eq(target.view_as(pred)).sum().item() 
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model,loss_func,dataset_dl,opt=None):
    
    run_loss=0.0 
    t_metric=0.0
    len_data=len(dataset_dl.dataset)

    
    for xb, yb in dataset_dl:
        
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb) 
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt) 
        run_loss+=loss_b        

        if metric_b is not None: 
            t_metric+=metric_b    
    
    loss=run_loss/float(len_data)  
    metric=t_metric/float(len_data) 
    
    return loss, metric

params_train={
 "train": train_dl,"val": val_dl,
 "epochs": 34,
 "optimiser": optim.Adam(cnn_model.parameters(),
                         lr=3e-4),
 "lr_change": ReduceLROnPlateau(opt,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=1),
 "f_loss": nn.NLLLoss(),  #reduction="sum"
 "weight_path": "weights.pt",
 "check": False, 
}



from tqdm.notebook import trange, tqdm

def train_val(model, params,verbose=True):
    
    
    epochs=params["epochs"]
    loss_func=params["f_loss"]
    opt=params["optimiser"]
    train_dl=params["train"]
    val_dl=params["val"]
    lr_scheduler=params["lr_change"]
    weight_path=params["weight_path"]
    
    loss_history={"train": [],"val": []} 
    metric_history={"train": [],"val": []} 
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_loss=float('inf') 
    
    
    for epoch in tqdm(range(epochs)):
        
       
        current_lr=get_lr(opt)
        if(verbose):
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
        
       
        
        model.train()
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,opt)

        
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
       
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func,val_dl)
        
       
        if(val_loss < best_loss):
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            
            torch.save(model.state_dict(), weight_path)
            if(verbose):
                print("Copied best model weights!")
        
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if(verbose):
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 

        if(verbose):
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
            print("-"*10) 

    
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

params_train={
 "train": train_dl,"val": val_dl,
 "epochs": 34,
 "optimiser": optim.Adam(cnn_model.parameters(),lr=3e-4), 
 "lr_change": ReduceLROnPlateau(opt,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=1),
 "f_loss": nn.NLLLoss(),   #reduction="sum"
 "weight_path": "weights.pt",
}



cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train)



import seaborn as sns; sns.set(style='whitegrid')

epochs=params_train["epochs"]

fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],ax=ax[0],label='loss_hist["train"]')
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],ax=ax[0],label='loss_hist["val"]')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["train"],ax=ax[1],label='metric_hist["train"]')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["val"],ax=ax[1],label='metric_hist["val"]')
plt.title('Convergence History')





class pytorchdata_test(Dataset):
    
    def __init__(self, data_dir, transform,data_type="train"):
        
        path2data = os.path.join(data_dir,data_type)
        filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        
        
        csv_filename="sample_submission.csv"
        path2csvLabels=os.path.join(data_dir,csv_filename)
        labels_df=pd.read_csv(path2csvLabels)
        
        
        labels_df.set_index("img_id", inplace=True)
        
        
        self.labels = [labels_df.loc[filename[:-5]].values[0] for filename in filenames]
        self.transform = transform       
        
    def __len__(self):
        
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.full_filenames[idx]) 
        image = self.transform(image)
        return image, self.labels[idx]





cnn_model.load_state_dict(torch.load('weights.pt'))


path_sub = "C:/Users/break/Desktop/project1/sample_submission.csv"
labels_df = pd.read_csv(path_sub)
labels_df.shape


data_dir = 'C:/Users/break/Desktop/project1/'

data_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((195,195))])

img_dataset_test = pytorchdata_test(data_dir,data_transformer,data_type="test")
print(len(img_dataset_test), 'samples found')



def inference(model,dataset,device,num_classes=2):
    
    len_data=len(dataset)
    y_out=torch.zeros(len_data,num_classes) 
    y_gt=np.zeros((len_data),dtype="uint8") 
    model=model.to(device) 
    
    with torch.no_grad():
        for i in tqdm(range(len_data)):
            x,y=dataset[i]
            y_gt[i]=y
            y_out[i]=model(x.unsqueeze(0).to(device))

    return y_out.numpy(),y_gt

y_test_out,_ = inference(cnn_model,img_dataset_test, device)


y_test_pred=np.argmax(y_test_out,axis=1)
print(y_test_pred.shape)
print(y_test_pred[0:5])


preds = np.exp(y_test_out[:, 1])
print(preds.shape)
print(preds[0:5])

url3 = "C:/Users/break/Desktop/project1/sample_submission.csv"
import pandas as pd
sample=pd.read_csv(url3)
sample['cancer_score']=preds
sample.to_csv("C:/Users/break/Desktop/project1/submission.csv")