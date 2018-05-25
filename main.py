import sys
from glob import glob

# whether your path is setting right?
pathCheck = input("Do you check path about preprocess files yet? ")
if pathCheck == 'yes':
    pass
else:
    sys.exit()

# point out the file to preprocess
setInput = input('Which data would you want to preprocess? (LUNA_anno, LUNA_cand, TanChi): ')

while setInput != 'LUNA_anno' and setInput != 'LUNA_cand' and setInput != 'TanChi':
    setInput = input('Please enter the specified input (LUNA_anno, LUNA_cand, TanChi): ')

if setInput == 'LUNA_anno':
    ## establish preprocess subset list
    import GenerateHit as GH
    
    List_subset = []
    for i in range(0,10):
        tmp = 'subset' + str(i)
        List_subset.append(tmp)

    ## preprocess tumor image data
    for i in range(0, len(List_subset)):
        Path_subset, List_filenames = GH.GetName(GH.Path_data, List_subset[i])
        df_tumor = GH.TumorLocate(Path_subset, List_filenames, GH.Path_data)
        GH.TumorImg(Path_subset, List_filenames, df_tumor)

elif setInput == 'LUNA_cand':
    import GenerateAlarm as GA
    
    ## establish preprocess subset list
    List_subset = []
    for i in range(0,10):
        tmp = 'subset' + str(i)
        List_subset.append(tmp)

    ## preprocess candidate image data
    for i in range(0, len(List_subset)):
        Path_subset, List_filenames = GA.GetName(GA.Path_data, List_subset[i])
        df_candidate = GA.CandidateLocate(Path_subset, List_filenames, GA.Path_data)
        GA.CandidateImg(Path_subset, List_filenames, df_candidate)

elif setInput == 'TanChi':
    import GenerateTanChi as GT
    
    ## establish preprocess subset list
    List_subset = []
    for i in range(0, 15):
        tmp = ('train_subset%02d' % i)
        List_subset.append(tmp)

    ## preprocess candidate image data
    for i in range(0, len(List_subset)):
        Path_subset, List_filenames = GT.GetName(GT.Path_data, List_subset[i])
        df_tumor = GT.TumorLocate(Path_subset, List_filenames, GT.Path_data)
        GT.TumorImg(Path_subset, List_filenames, df_tumor)
