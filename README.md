## rocket dataset
本数据集测试了两种方式，一种是平移的测试，一种是旋转的测试；在文件夹主要包含match和match_rotate两个数据集，match是平移数据集，natch_rotate是旋转数据集。每个文件夹里有两种图片，一种是sar,一种是optical,mat.txt文件为optical到sar的变换矩阵。
在dataset/__init__文件里配置测试文件sar.txt
```
        train_data_file="/four_disk/image_patch_dataset//stage1/train/train.txt"
        test_data_file="/four_disk/image_patch_dataset//stage1/sar.txt"
        return build_gl(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
```
在dataset/dataset2.py里修改导入文件
```
        mat = np.loadtxt(os.path.dirname(sar_img_path) + '/mat.txt',  delimiter=',')
        mat = np.concatenate([mat,np.array([[0.,0.,1.]],np.float64)])
```
默认为平移变换


 
