import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob

## set data path
Path_data = '/home/mcas/FTP/TCW/TanChi/'
Path_save = '/home/mcas/TCW/preprocessed/'

## subset name and files name in subset
def GetName(Path_data, subset):
    '''
    Input:
      Path_data: path to all subset directory. (string)
      subset: the subset that you specified. (string)
    Output:
      Path_subset: path of specified subset directory. (string)
      List_filenames: all files name in this subset. (list)
    '''
    Path_subset = Path_data + subset + '/'
    List_filenames = glob(Path_subset + '*.mhd')
    return Path_subset, List_filenames

## read tumor location data
def TumorLocate(Path_subset, List_filenames, Path_data):
    '''
    Input:
      Path_subset: path of specified subset directory. (string)
      List_filenames: all files name in this subset. (list)
      Path_data: path to all subset directory. (string)
    Output:
      df_tumor: all tumor information of this subset(pd.dataframe)
    '''
    df_tumor = pd.read_csv(Path_data + 'annotations.csv')
    df_tumor['seriesuid'] = Path_subset + df_tumor['seriesuid'] + '.mhd'
    df_tumor['file'] = 0
    for i in range(0, len(List_filenames)):
        tmp_count = (df_tumor['seriesuid'] == List_filenames[i])
        df_tumor['file'] = df_tumor['file'] + (tmp_count*1)
    df_tumor = df_tumor[:][df_tumor['file'] == 1]
    return df_tumor

## extract tumor image from whole mhd file (with this step, we can downscale input)
def TumorImg(Path_subset, List_filenames, df_tumor):
    '''
    Input:
      Path_subset: path of specified subset directory. (string)
      List_filenames: all files name in this subset. (list)
      df_tumor: all tumor information of this subset(pd.dataframe)
    Output:
      None, but save tumor image and tumor position at Path_save
    '''
    for fcount, ImgFile in enumerate(List_filenames):
        print('preprocess on ' + ImgFile)
        DF_mini_tumor = df_tumor[df_tumor['seriesuid'] == ImgFile]
        if DF_mini_tumor.shape[0] > 0:
            itk_img = sitk.ReadImage(ImgFile)
            Array_img = sitk.GetArrayFromImage(itk_img)    # indexes are z, y, x
            num_z, height, width = Array_img.shape         # take tumor positon
            origin = np.array(itk_img.GetOrigin())         # take mhd origin
            spacing = np.array(itk_img.GetSpacing())       # take voxel size
            for idx_node, idx_row in DF_mini_tumor.iterrows():
                # take tumor position information
                node_x = idx_row['coordX']
                node_y = idx_row['coordY']
                node_z = idx_row['coordZ']
                diam = idx_row["diameter_mm"]
                # take only tumor image with above information
                center = np.array([node_x, node_y, node_z])
                v_center = np.rint( (center-origin)/spacing )
                img = Array_img[int(v_center[2])]
                # normalize the img
                img_norm = (img-np.min(img))/(np.max(img)-np.min(img))
                np.save(Path_save + 'Imgs_t_%04d_%06d.npy' % (fcount, idx_node), img_norm)
                tmp_ImgName = 'Imgs_t_%04d_%06d.npy' % (fcount, idx_node)
                saveInfo = np.array([tmp_ImgName, node_x, node_y, node_z, diam, 1])
                np.savetxt(Path_save + 'Labs_t_%04d_%06d.csv' % (fcount, idx_node), saveInfo, fmt = '%s', newline = ',')
