from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

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

def load_image(df_features, transform, datapath1, datapath2):
    transformed_images = []
    transformed_features = []
    transformed_features2 = []
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
                
                fft_matrix = fft_calculation(image_copy, size=96, step=8)
                fft_matrix = cv2.resize(fft_matrix, (224, 224))
                fft_matrix = (fft_matrix - np.min(fft_matrix)) / (np.max(fft_matrix) - np.min(fft_matrix))
                
                clr_matrix = np.load(os.path.join('/vinserver_user/khietdang/data/datasetA/clr_npy', filename).split('.')[0] + 'Y.npy')
                clr_matrix = cv2.resize(clr_matrix, (224, 224))
                clr_matrix = clr_matrix / 5
                
                gX = cv2.Sobel(image_copy, cv2.CV_64F, 1, 0)
                gY = cv2.Sobel(image_copy, cv2.CV_64F, 0, 1)
                magnitude = np.sqrt((gX ** 2) + (gY ** 2))
                magnitude = magnitude / (magnitude.max() - magnitude.min())
                magnitude = cv2.resize(magnitude_norm, (224,224))
                
                #feature_matrix = np.stack((magnitude, magnitude, magnitude, magnitude), axis=0)
                feature_matrix2 = np.stack((fft_matrix, clr_matrix, magnitude), axis=0)
                feature_matrix = np.stack((fft_matrix, clr_matrix, magnitude, magnitude), axis=0)

                image = np.repeat(np.expand_dims(image, axis=2), repeats=3, axis=2)
                image = transform(image)
                transformed_images.append(image)

                feature_matrix = torch.tensor(feature_matrix)
                feature_matrix2 = torch.tensor(feature_matrix2)
                transformed_features.append(feature_matrix.float())
                transformed_features2.append(feature_matrix2.float())

    return torch.stack(transformed_images), torch.stack(transformed_features)

def process_csv(csvpath1, csvpath2):
    df_feats_1 = pd.read_csv(csvpath1)
    df_feats_2 = pd.read_csv(csvpath2)

    # Dataset 1 had 0 scores for cells to indicate absence of structure/gfp
    df_feats_1["gfp_keep"] = (df_feats_1.kg_structure_org_score > 0) & (df_feats_1.mh_structure_org_score > 0)

    # Dataset 2 scored afterword; 0 in "no_structure" indicates absence of structure/gfp in cell; all other cells are NaN
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
