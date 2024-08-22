import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
import time
from scipy.stats import spearmanr
from utils import load_image, process_csv, process_meta
from model import DSarcNet
random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--csvpath1', type=str, required=True)
parser.add_argument('--csvpath2', type=str, required=True)
parser.add_argument('--metapath1', type=str, required=True)
parser.add_argument('--metapath2', type=str, required=True)
parser.add_argument('--datapath1', type=str, required=True)
parser.add_argument('--datapath2', type=str, required=True)
parser.add_argument('--cuda', type=int,default=0)
parser.add_argument('--numworkers', type=int, default=3)
parser.add_argument('--outpath', type=str, default='./output')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

NUM_WORKERS = args.numworkers

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_TRUES = os.path.join(PATH, 'test_trues.log')
TEST_BOTH =  os.path.join(PATH, 'test_both.log')

BATCH_SIZE = args.batch_size
  
all_fish_df = process_csv(args.csvpath1, args.csvpath2)
all_gs_df = process_meta(args.metapath1, args.metapath2)

fish_df = pd.merge(
        left=all_fish_df,
        right=all_gs_df,
        on=["napariCell_ObjectNumber", "fov_path"],
        how="inner",
    )

metadata_cols_in = [
    "napariCell_ObjectNumber",
    "rescaled_2D_single_cell_tiff_path",
    "fov_path",
    "cell_age",
    "mh_structure_org_score",
    "kg_structure_org_score",
]

fish_df = fish_df[metadata_cols_in]

# Calculate mean expert score
fish_df["Expert structural annotation score (average)"] = fish_df[
    ["mh_structure_org_score", "kg_structure_org_score"]
].mean(axis="columns")

assert len(fish_df) == len(
    fish_df[["napariCell_ObjectNumber", "fov_path"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

##### LINEAR REGRESSION #####

# columns / facet definitons and order
BAR_PLOT_COLUMNS = [
    "rescaled_2D_single_cell_tiff_path",
]

### Dataframe and model
all_good_scores = (fish_df.kg_structure_org_score > 0) & (
        fish_df.mh_structure_org_score > 0) & (np.abs(fish_df.kg_structure_org_score - fish_df.mh_structure_org_score) <= 1)
fish_df = fish_df[all_good_scores].reset_index(drop=True) #(5761,18)
print(fish_df.shape)

train, test = train_test_split(fish_df, test_size=0.2, random_state=42, stratify=fish_df['mh_structure_org_score'])
train, val = train_test_split(train, test_size=0.2, random_state=42, stratify=train['mh_structure_org_score'])

feat_cols = [c for c in BAR_PLOT_COLUMNS if c in fish_df.columns]

#DataFrame for Machine Learning (Linear Regression)
df_train = fish_df.loc[train.index, feat_cols + ["Expert structural annotation score (average)"]].copy()
df_test = fish_df.loc[test.index, feat_cols + ["Expert structural annotation score (average)"]].copy()
df_val = fish_df.loc[val.index, feat_cols + ["Expert structural annotation score (average)"]].copy()

custom_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    ]
)

print('---> Load & Transform Images start')

transformed_train_X, transformed_train_features = load_image(df_train, custom_transform, args.datapath1, args.datapath2, DEVICE)
transformed_test_X, transformed_test_features = load_image(df_test, custom_transform, args.datapath1, args.datapath2, DEVICE)
transformed_val_X, transformed_val_features = load_image(df_val, custom_transform, args.datapath1, args.datapath2, DEVICE)

print('---> Load & Transform Images done')

print(transformed_train_X.shape) 

train_dataset = []
for i in range(df_train.shape[0]):
    tensor1 = transformed_train_X[i]
    tensor2 = transformed_train_features[i]
    tensor3 = df_train["Expert structural annotation score (average)"].tolist()[i] 
    train_dataset.append((tensor1, tensor2, tensor3))

test_dataset = []
for i in range(df_test.shape[0]):
    tensor1 = transformed_test_X[i]
    tensor2 = transformed_test_features[i]
    tensor3 = df_test["Expert structural annotation score (average)"].tolist()[i]
    test_dataset.append((tensor1, tensor2, tensor3))
    
val_dataset = []
for i in range(df_val.shape[0]):
    tensor1 = transformed_val_X[i]
    tensor2 = transformed_val_features[i]
    tensor4 = df_val["Expert structural annotation score (average)"].tolist()[i]
    val_dataset.append((tensor1, tensor2, tensor3))

#DataLoader
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        )

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS)

print(len(train_loader), len(test_loader), len(val_loader))

print('---> DataLoader done')

# ##########################
# # MODEL

# Loss function
loss_fn = nn.MSELoss()

model = DSarcNet(device=DEVICE)
model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

start_time = time.time()

print('---> Training Start')
best_mse, best_epoch, best_corr, best_epoch2, best_r2, best_epoch3 = 999, -1, -1, -1, -100, -1
loss_train = []
loss_test = []
for epoch in range(args.num_epochs):
    losses = []
    model.train()
    for batch_idx, (images, den_matrix, targets) in enumerate(train_loader):
        den_matrix = den_matrix.to(DEVICE)
        images = images.to(DEVICE)
        targets = targets.float().to(DEVICE)

        # FORWARD AND BACK PROP
        output = model(images, den_matrix)

        loss = loss_fn(torch.squeeze(output), targets)
        losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 20:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                % (epoch+1, args.num_epochs, batch_idx,
                    len(train_dataset)//BATCH_SIZE, loss))
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)
    loss_train.append(np.mean(losses))
    
    model.eval()
    all_true_test = []
    all_pred_test = []
    with torch.set_grad_enabled(False):
        num_examples = 0
        test_loss_sum = 0
        for batch_idx, (images, den_matrix, targets) in enumerate(val_loader):
            den_matrix = den_matrix.to(DEVICE)
            images = images.to(DEVICE)
            targets = targets.float().to(DEVICE)
    
            # FORWARD AND BACK PROP
            output = model(images, den_matrix)

            test_loss = loss_fn(torch.squeeze(output), targets)
        
            losses.append(test_loss.item())

            test_loss_sum += test_loss.item()

            output = output.squeeze()

            all_pred_test.extend(output.tolist())

            lst_true = [str(float(i)) for i in targets]
            all_true_test.extend(lst_true)
        
        print(test_loss_sum)
        average_test_loss = test_loss_sum / len(test_loader)
        loss_test.append(average_test_loss)


        all_pred_float = [float(pred) for pred in all_pred_test] 
        all_true_float = [float(true) for true in all_true_test]

        # Calculate Spearman correlation
        spearman_corr, _ = spearmanr(all_true_float, all_pred_float)
        r2 = r2_score(all_true_float, all_pred_float)

        if average_test_loss < best_mse:
            best_mse, best_epoch = average_test_loss, epoch
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model_loss.pt'))

        if spearman_corr > best_corr:
            best_corr, best_epoch2 = spearman_corr, epoch
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model_corr.pt'))
            
        if r2 > best_r2:
            best_r2, best_epoch3 = r2, epoch
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model_r2.pt'))

        s = 'MAE/RMSE: | Current Valid: %.3f| Corr: %.3f Ep. %d | Best Valid : %.3f Ep. %d | Best Corr : %.3f Ep. %d | Best R2 : %.3f Ep. %d' % (
            average_test_loss, spearman_corr, epoch, best_mse, best_epoch, best_corr, best_epoch2, best_r2, best_epoch3)
        print(s)
    
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)
        
    plt.plot(np.array(loss_train), 'b')
    plt.plot(np.array(loss_test), 'orange')
    plt.savefig(os.path.join(PATH, 'loss.png'))

########## EVALUATE BEST MODEL ######
print('Best model loss')
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model_loss.pt')))
model.eval()

########## SAVE PREDICTIONS ######
all_true = []
all_pred = []
all_cnn = []
with torch.set_grad_enabled(False):
    for batch_idx, (images, den_matrix, targets) in enumerate(test_loader):
        den_matrix = den_matrix.to(DEVICE)
        images = images.to(DEVICE)
        targets = targets.float().to(DEVICE)

        output = model(images, den_matrix)
        
        if output.shape[0] != 1:
            output = output.squeeze()
            all_pred.extend(output.tolist())
        else:
            all_pred.extend([output])

        lst_true = [str(float(i)) for i in targets]
        all_true.extend(lst_true)

all_pred_float = np.array([float(pred) for pred in all_pred]) 
all_true_float = np.array([float(true) for true in all_true])
all_cnn_float = np.array([float(cnn) for cnn in all_cnn])

spearman_corr, _ = spearmanr(all_true_float, all_pred_float)
mae = np.sum(np.abs(all_true_float - all_pred_float)) / len(all_pred_float)
r2 = r2_score(all_true_float, all_pred_float)
mse = np.sum((all_true_float - all_pred_float) ** 2) / len(all_pred_float)

print(f'Spearman Correlation: {spearman_corr:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R2 score: {r2:.4f}')

########## EVALUATE BEST MODEL ######
print('Best model correlation')
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model_corr.pt')))
model.eval()

########## SAVE PREDICTIONS ######
all_true = []
all_pred = []
all_cnn = []
with torch.set_grad_enabled(False):
    for batch_idx, (images, den_matrix, targets) in enumerate(test_loader):
        den_matrix = den_matrix.to(DEVICE)
        images = images.to(DEVICE)
        targets = targets.float().to(DEVICE)

        output = model(images, den_matrix)
        
        if output.shape[0] != 1:
            output = output.squeeze()
            all_pred.extend(output.tolist())
        else:
            all_pred.extend([output])

        lst_true = [str(float(i)) for i in targets]
        all_true.extend(lst_true)

all_pred_float = np.array([float(pred) for pred in all_pred])
all_true_float = np.array([float(true) for true in all_true])
all_cnn_float = np.array([float(cnn) for cnn in all_cnn])

spearman_corr, _ = spearmanr(all_true_float, all_pred_float)
mae = np.sum(np.abs(all_true_float - all_pred_float)) / len(all_pred_float)
r2 = r2_score(all_true_float, all_pred_float)
mse = np.sum((all_true_float - all_pred_float) ** 2) / len(all_pred_float)

print(f'Spearman Correlation: {spearman_corr:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R2 score: {r2:.4f}')

########## EVALUATE BEST MODEL ######
print('Best model r2')
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model_r2.pt')))
model.eval()

########## SAVE PREDICTIONS ######
all_true = []
all_pred = []
all_cnn = []
with torch.set_grad_enabled(False):
    for batch_idx, (images, den_matrix, targets) in enumerate(test_loader):
        den_matrix = den_matrix.to(DEVICE)
        images = images.to(DEVICE)
        targets = targets.float().to(DEVICE)

        output = model(images, den_matrix)
        
        if output.shape[0] != 1:
            output = output.squeeze()
            all_pred.extend(output.tolist())
        else:
            all_pred.extend([output])

        lst_true = [str(float(i)) for i in targets]
        all_true.extend(lst_true)

all_pred_float = np.array([float(pred) for pred in all_pred])
all_true_float = np.array([float(true) for true in all_true])
all_cnn_float = np.array([float(cnn) for cnn in all_cnn])

spearman_corr, _ = spearmanr(all_true_float, all_pred_float)
mae = np.sum(np.abs(all_true_float - all_pred_float)) / len(all_pred_float)
r2 = r2_score(all_true_float, all_pred_float)
mse = np.sum((all_true_float - all_pred_float) ** 2) / len(all_pred_float)

print(f'Spearman Correlation: {spearman_corr:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R2 score: {r2:.4f}')
