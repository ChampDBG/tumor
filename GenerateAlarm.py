import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob

## set data path
Path_data = '/home/mcas/FTP/TCW/LUNA16/'
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

## read candidates location data
def CandidateLocate(Path_subset, List_filenames, Path_data):
    '''
    Input:
      Path_subset: path of specified subset directory. (string)
      List_filenames: all files name in this subset. (list)
      Path_data: path to all subset directory. (string)
    Output:
      df_candidate: all candidate information of this subset(pd.dataframe)
    '''
    df_candidate = pd.read_csv(Path_data + 'candidates.csv')
    df_candidate['seriesuid'] = Path_subset + df_candidate['seriesuid'] + '.mhd'
    df_candidate['file'] = 0
    for i in range(0, len(List_filenames)):
        tmp_count = (df_candidate['seriesuid'] == List_filenames[i])
        df_candidate['file'] = df_candidate['file'] + (tmp_count*1)
    df_candidate = df_candidate[:][df_candidate['file'] == 1]
    df_candidate = df_candidate[:][df_candidate['class'] == 0]
    return df_candidate

## extract candidate image from whole mhd file (with this step, we can downscale input)
def CandidateImg(Path_subset, List_filenames, df_candidate):
    '''
    Input:
      Path_subset: path of specified subset directory. (string)
      List_filenames: all files name in this subset. (list)
      df_candidate: all candidate information of this subset(pd.dataframe)
    Output:
      None, but save candidate image and candidate position at Path_save
    '''
    for fcount, ImgFile in enumerate(List_filenames):
        print('preprocess on ' + ImgFile)
        DF_mini_candidate = df_candidate[df_candidate['seriesuid'] == ImgFile]
        if DF_mini_candidate.shape[0] > 0:
            itk_img = sitk.ReadImage(ImgFile)
            Array_img = sitk.GetArrayFromImage(itk_img)    # indexes are z, y, x
            num_z, height, width = Array_img.shape         # take candidate positon
            origin = np.array(itk_img.GetOrigin())         # take mhd origin
            spacing = np.array(itk_img.GetSpacing())       # take voxel size
            for idx_node, idx_row in DF_mini_candidate.iterrows():
                # take candidate position information
                node_x = idx_row['coordX']
                node_y = idx_row['coordY']
                node_z = idx_row['coordZ']
                # take only candidate image with above information
                center = np.array([node_x, node_y, node_z])
                v_center = np.rint( (center-origin)/spacing )
                img = Array_img[int(v_center[2])].copy()
                img = img.astype(float)
                # background setting
                mask_floor = img < -600
                mask_ceil = img > 100
                mask = mask_floor | mask_ceil
                img[mask] = -2048
                # normalize the img
                Nnum = img[img > -2048]
                img[img > -2048] = (Nnum - np.min(Nnum)) / ( np.max(Nnum) - np.min(Nnum) )
                img[img == -2048] = -1
                np.save(Path_save + 'Imgs_c_%04d_%06d.npy' % (fcount, idx_node), img)
                tmp_ImgName = 'Imgs_c_%04d_%06d.npy' % (fcount, idx_node)
                saveInfo = np.array([tmp_ImgName, None, None, None, None, 0])
                np.savetxt(Path_save + 'Labs_c_%04d_%06d.csv' % (fcount, idx_node), saveInfo, fmt = '%s', newline = ',')
