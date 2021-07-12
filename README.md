# 3DResUNet-DSN-CRF-Liver-segmentation



LITS17-challenge https://competitions.codalab.org/competitions/17094#results


How to use it?



step1 python preprocess.py(先在parameter.py中配置文件路径)

      data
         origin-data
              ct
               volume-0.nii
               ......
              seg
               segmentation-0.nii
               ......
         fixed-data
              ct
               volume-0.nii
               ......
              seg
               segmentation-0.nii
               ......
    


step2 python main.py  


step3 python test.py 


step4  python 2D-CRF.py or 3D-CRF.py

visual result: MITK、ITK-SNAP、Seg3D
