
import numpy as np 
import joblib
import glob as gb 

import torch
from model import my_model 
from model import Essentiality_Dataset as Data
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")










imgs_dir = '/work3/sajata/ACB-course/cancer-essential-gene/data/imgs_100_genes/'
lbls_dir = '/work3/sajata/ACB-course/cancer-essential-gene/data/labels_100_genes.joblib'

imgs_path = gb.glob(imgs_dir+'*.npy')
labels = joblib.load(lbls_dir)


dataset = Data(imgs_path, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)



model = my_model().to(device)

criterion = torch.nn.MSELoss().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, torch.tensor(0.001))

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    criterion.train()
    running_loss = 0.0
    counter = 0
    for imgs, lbls in train_loader:
        imgs = torch.stack(imgs).to(device)
        preds = model(imgs)
        lbls = torch.stack(lbls).to(device)
        lbls = lbls.reshape(shape=(lbls.shape[0], 1))
        loss = criterion(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # running_loss += loss.item()
        # counter+=1
        #print(counter, 'loss: ', loss.item())
        #if counter==100:
         #   break 
    lr_scheduler.step()
    
    
    model.eval()
    criterion.eval()
    pred_all = []
    real_all = []
    for imgs, lbls in train_loader:
        imgs = torch.stack(imgs).to(device)
        preds = model(imgs)
        pred_all.extend(preds.reshape(-1).to('cpu').detach().numpy())
        real_all.extend([lbl.to('cpu').detach().numpy() for lbl in lbls])
        lbls = torch.stack(lbls).to(device)
        lbls = lbls.reshape(shape=(lbls.shape[0], 1))
        #loss = criterion(preds, lbls)
        #running_loss += loss.item()
        #counter+=1
        
    real_all = np.array(real_all).reshape(-1)
    pred_all = np.array(pred_all).reshape(-1)
    r2_train = r2_score(real_all, pred_all)
    pred_all = torch.tensor(pred_all)
    real_all = torch.tensor(real_all)
    loss_train = criterion(pred_all, real_all)

    #epoch_loss = running_loss / counter #len(dataset)
    msg = f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_train:.4f}, r2: {r2_train:.4f}'
    print(msg)
    f = open('results.txt', mode='a+')
    f.write(msg+'\n')
    f.close()
    


