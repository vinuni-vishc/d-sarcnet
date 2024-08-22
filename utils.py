from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import copy
import math
import torch
from scipy import ndimage
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.measure import block_reduce
from scipy.ndimage.morphology import distance_transform_edt as edt

def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def __get_patch__(image, location, window_size):
        xCoor = int(location[0])
        yCoor = int(location[1])
        imShape = np.shape(image)
        image_crop = np.zeros((window_size, window_size), dtype=np.float16)
        crop_center = math.ceil(window_size / 2)

        if window_size % 2 == 0:
            xMinus = window_size / 2
            xPlus = xMinus
            yMinus = window_size / 2
            yPlus = yMinus
        else:
            xMinus = xPlus = window_size // 2
            xMinus = yPlus = window_size // 2

        xMinus = int(np.min([xMinus, xCoor]))
        yMinus = int(np.min([yMinus, yCoor]))
        xPlus = int(np.min([xPlus, imShape[0] - xCoor]))
        yPlus = int(np.min([yPlus, imShape[1] - yCoor]))

        image_crop[
            (crop_center - xMinus) : (crop_center + xPlus),
            (crop_center - yMinus) : (crop_center + yPlus),
        ] = image[
            (xCoor - xMinus) : (xCoor + xPlus), (yCoor - yMinus) : (yCoor + yPlus)
        ]

        return image_crop

def fft_cal(im):
    fft = np.fft.fft2(im)
    return np.sum(np.abs(fft) ** 2)

def fft_calculation(im, size, step):
    fft_matrix = np.zeros((im.shape[0] // step + 1, im.shape[1] // step + 1))
    for i in range(0, im.shape[0], step):
        for j in range(0, im.shape[1], step):
            im_copy = copy.deepcopy(im)
            im_dat = __get_patch__(im_copy, (i, j), window_size=size)
            fft_matrix[int(i // step), int(j // step)] = fft_cal(im_dat)
    
    return fft_matrix

def preprocess(im, gaussian_filter_size=1):
    im = cv2.Laplacian(im,cv2.CV_64F)
    im = ndimage.gaussian_filter(im, gaussian_filter_size)
    
    return im

class local_convnext_model(nn.Module):
    def __init__(self, num_classes=5):
        super(local_convnext_model, self).__init__()

        # resnet18 = models.resnet18(pretrained=True)
        resnet18 = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        res_modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*res_modules)
        
        self.cnn_3x3_conv = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        self.cnn_1x1_conv = nn.Conv2d(768, 768, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.norm2 = nn.LayerNorm(768)
        
        self.fc = self.fcClassifier(768, num_classes)

    def forward(self, x):
        x = F.interpolate(x, (224, 224))
        x = torch.cat((x,x,x),dim=1)
        
        x = self.resnet18(x)
        x = self.cnn_3x3_conv(x)
        x = self.cnn_1x1_conv(x)        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.norm2(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, numClasses),
        )

        return fc_classifier

class local_pattern:
    def __init__(self, model_path):
        super(local_pattern, self).__init__()

        self.__model = local_convnext_model()
        self.__model.load_state_dict(torch.load(model_path))  # loading trained local pattern model
        self.__tf_raw = transforms.Compose(
            [  
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0966], std=[0.1153]),
            ]
        )
        self.__tf_aug = transforms.Compose(
            [  
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0966], std=[0.1153]),
            ]
        )
        self.__window_size = 96  # size of local patch used for classification
    

    def predict_image_sliding(self, raw_im, data_mask, stride, batch_size, interp_order, device, use_resize=True):
    
        self.__model.to(device).eval()
        im_shape = np.shape(raw_im)
        prob_map = np.zeros(
            (5, im_shape[0] // stride, im_shape[1] // stride)
        )  
        raw_im = raw_im.astype(np.float32)
        raw_im = (raw_im - np.min(raw_im)) / (np.max(raw_im) - np.min(raw_im))

        curr_batch_0 = None
        curr_batch_1 = None
        curr_batch_2 = None
        curr_batch_3 = None
        batch_r = []
        batch_c = []
        in_batch = 0

        last_r = range(0, im_shape[0], stride)[-1]
        last_c = range(0, im_shape[1], stride)[-1]

        for r in tqdm(range(0, im_shape[0], stride)):
            for c in range(0, im_shape[1], stride):
                if data_mask[r, c] > 0:
                    continue
                # build input batch
                if (in_batch < (batch_size)) and (r < last_r and c < last_c):
                    patch = self.__get_patch__(raw_im, (r, c))
                    batch_r.append(r // stride)
                    batch_c.append(c // stride)

                    if in_batch == 0:
                        patch_0 = self.__tf_raw.__call__(torch.Tensor(patch)).to(device)
                        patch_1 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_2 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_3 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)

                        curr_batch_0 = patch_0[None, :, :, :]
                        curr_batch_1 = patch_1[None, :, :, :]
                        curr_batch_2 = patch_2[None, :, :, :]
                        curr_batch_3 = patch_3[None, :, :, :]
                    else:
                        patch_0 = self.__tf_raw.__call__(torch.Tensor(patch)).to(device)
                        patch_1 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_2 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_3 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)

                        curr_batch_0 = torch.cat(
                            (curr_batch_0, patch_0[None, :, :, :]), dim=0
                        )
                        curr_batch_1 = torch.cat(
                            (curr_batch_1, patch_1[None, :, :, :]), dim=0
                        )
                        curr_batch_2 = torch.cat(
                            (curr_batch_2, patch_2[None, :, :, :]), dim=0
                        )
                        curr_batch_3 = torch.cat(
                            (curr_batch_3, patch_3[None, :, :, :]), dim=0
                        )

                    in_batch += 1

                # evaluate batches with model and average predictions
                if (in_batch == batch_size or (r == last_r and c == last_c)) and curr_batch_0 is not None:
                    predictions = (
                        self.__model(curr_batch_0).cpu().detach().numpy().squeeze() / 4
                    )
                    predictions += (
                        self.__model(curr_batch_1).cpu().detach().numpy().squeeze() / 4
                    )
                    predictions += (
                        self.__model(curr_batch_2).cpu().detach().numpy().squeeze() / 4
                    )
                    predictions += (
                        self.__model(curr_batch_3).cpu().detach().numpy().squeeze() / 4
                    )

                    if len(predictions.shape) == 1:
                        predictions = predictions[None, :]
                    predictions = (
                        F.softmax(torch.Tensor(predictions), dim=1).numpy()
                    )
                    for batch_index in range(len(batch_r)):
                        br = batch_r[batch_index]
                        bc = batch_c[batch_index]
                        for bclass in range(5):
                            prob_map[bclass, br, bc] = predictions[batch_index, bclass]

                    curr_batch_0 = None
                    curr_batch_1 = None
                    curr_batch_2 = None
                    curr_batch_3 = None
                    batch_r = []
                    batch_c = []
                    in_batch = 0
        if use_resize:
            prob_map = resize(prob_map, (5, im_shape[0], im_shape[1]), order=interp_order)

        return prob_map

    def __get_patch__(self, image, location):
        ##
        # Gets cropped patch of image at given location. Images are zero-padded for edge-cases
        ##
        window_size = self.__window_size
        xCoor = int(location[0])
        yCoor = int(location[1])
        imShape = np.shape(image)
        image_crop = np.zeros((window_size, window_size), dtype=np.float16)
        crop_center = math.ceil(window_size / 2)

        if window_size % 2 == 0:
            xMinus = window_size / 2
            xPlus = xMinus
            yMinus = window_size / 2
            yPlus = yMinus
        else:
            xMinus = xPlus = window_size // 2
            xMinus = yPlus = window_size // 2

        xMinus = int(np.min([xMinus, xCoor]))
        yMinus = int(np.min([yMinus, yCoor]))
        xPlus = int(np.min([xPlus, imShape[0] - xCoor]))
        yPlus = int(np.min([yPlus, imShape[1] - yCoor]))

        image_crop[
            (crop_center - xMinus) : (crop_center + xPlus),
            (crop_center - yMinus) : (crop_center + yPlus),
        ] = image[
            (xCoor - xMinus) : (xCoor + xPlus), (yCoor - yMinus) : (yCoor + yPlus)
        ]
        max_im = np.amax(image_crop)
        if max_im != 0:
            image_crop = image_crop / np.amax(image_crop)

        return image_crop

def local_pattern_calculation(image, device):
    local_pattern_path = './local_pattern_model.pth'
    classifier = local_pattern(model_path=local_pattern_path) 
    step = 8
    
    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    data_binary = preprocess(image)

    data_binary = (data_binary - np.min(data_binary)) / (np.max(data_binary) - np.min(data_binary))
    thres = threshold_otsu(data_binary)
    data_binary[data_binary >= thres] = 1
    data_binary[data_binary < thres] = 0
    data_binary = 1 - data_binary

    bkrad = 64
    data_mask = data_binary.copy()
    data_mask = data_mask.astype(np.uint16)
    data_tmp = block_reduce(data_mask, (bkrad, bkrad), np.sum)
    data_mask = (
        resize(
            data_tmp,
            data_mask.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
            mode="reflect",
        )
        < bkrad
    ).astype(np.uint8)
    data_label_fine = label(data_mask)
    data_label_coarse = label(edt(data_mask) > bkrad)
    for lb_c in np.unique(data_label_coarse.reshape(-1)):
        if lb_c > 0:
            y, x = np.where(data_label_coarse == lb_c)
            lb_f = np.unique(data_label_fine[y, x])
            data_label_fine[data_label_fine == lb_f] = -1
    data_mask = (data_label_fine < 0).astype(np.uint8)
    
    # CNN inference
    data_probs = classifier.predict_image_sliding(
        raw_im=image, data_mask=data_mask, stride=step, batch_size=120, interp_order=1, device=device, use_resize=False
    )
    data_mask = resize(data_mask, (data_mask.shape[0] // step, data_mask.shape[1] // step), order=1)
    data_mask = data_mask / np.max(data_mask)
    data_mask = np.round(data_mask) 

    # Reordering the classes to
    # 1 - Diffuse/others
    # 2 - Fibers
    # 3 - Disorganized puncta
    # 4 - Organized Puncta
    # 5 - Organized Z-disks

    data_classification = np.argmax(data_probs, axis=0) + 1
    data_classification = data_classification.astype(np.uint8)
    data_probs[:, data_mask > 0] = 0
    data_classification[data_mask > 0] = 0

    return data_classification

def load_image(df_features, transform, datapath1, datapath2, device):
    transformed_images = []
    transformed_features = []
    for path in df_features['rescaled_2D_single_cell_tiff_path']:
        filename = os.path.basename(path)
        local_image_path = os.path.join(datapath1, filename)
        if not os.path.exists(local_image_path):
              local_image_path = os.path.join(datapath2, filename)
        if local_image_path:
            if os.path.exists(local_image_path):  
                images = read_tiff(local_image_path)
                image = images[1]
                image_copy = image.copy()

                image = cv2.resize(image, (224,224)) #uint8
                
                #fft_matrix = np.load(os.path.join('./fft_power_images/', filename) + '.npy')
                fft_matrix = fft_calculation(image, size=96, step=8)
                fft_matrix = cv2.resize(fft_matrix, (224, 224))
                fft_matrix = (fft_matrix - np.min(fft_matrix)) / (np.max(fft_matrix) - np.min(fft_matrix))
                
 
                # local_pattern_matrix = np.load('.' + os.path.join('./local_patterns_images', filename).split('.')[1] + 'Y.npy')
                local_pattern_matrix = local_pattern_calculation(image, device)
                local_pattern_matrix = cv2.resize(local_pattern_matrix, (224, 224))
                local_pattern_matrix = local_pattern_matrix / 5
                
                gX = cv2.Sobel(image_copy, cv2.CV_64F, 1, 0)
                gY = cv2.Sobel(image_copy, cv2.CV_64F, 0, 1)
                magnitude = np.sqrt((gX ** 2) + (gY ** 2))
                magnitude = magnitude / (magnitude.max() - magnitude.min())
                magnitude = cv2.resize(magnitude, (224,224))
                
                feature_matrix = np.stack((fft_matrix, local_pattern_matrix, magnitude, magnitude), axis=0)

                image = np.repeat(np.expand_dims(image, axis=2), repeats=3, axis=2)
                image = transform(image)
                transformed_images.append(image)

                feature_matrix = torch.tensor(feature_matrix)
                transformed_features.append(feature_matrix.float())

    return torch.stack(transformed_images), torch.stack(transformed_features)

def process_csv(csvpath1, csvpath2):
    df_feats_1 = pd.read_csv(csvpath1)
    df_feats_2 = pd.read_csv(csvpath2)

    df_feats_1["gfp_keep"] = (df_feats_1.kg_structure_org_score > 0) & (df_feats_1.mh_structure_org_score > 0)
    df_feats_2["gfp_keep"] = df_feats_2.no_structure.isna()

    all_fish_df = pd.concat(
            [df_feats_1, df_feats_2]
        ).reset_index(drop=True)

    all_fish_df.napariCell_ObjectNumber = all_fish_df.napariCell_ObjectNumber.astype(int).astype(str)
    all_fish_df["Type"] = "FISH"
    return all_fish_df

def process_meta(metapath1, metapath2):
    df_gs_1 = pd.read_csv(metapath1)
    df_gs_2 = pd.read_csv(metapath2)

    all_gs_df = pd.concat(
            [df_gs_1, df_gs_2]
        ).reset_index(drop=True)

    all_gs_df.napariCell_ObjectNumber = all_gs_df.napariCell_ObjectNumber.astype(int).astype(str)

    all_gs_df = all_gs_df.rename(
        columns={"original_fov_location": "fov_path"}
    )
    all_gs_df = all_gs_df.drop(
        columns=["Age", "Dataset"]
    )
    
    return all_gs_df
