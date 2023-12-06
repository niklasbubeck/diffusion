import nibabel as nib 
import numpy as np 
import matplotlib.pyplot as plt 

test_load = nib.load("C://Users//nikla//CodingProjects//datasets//ACDC//database//training//patient001//patient001_frame01.nii.gz").get_fdata()
test_load.shape
test = test_load[:,:,59]
plt.imshow(test)
plt.show()