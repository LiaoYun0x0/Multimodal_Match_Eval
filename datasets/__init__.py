from datasets.dataset2 import build_gl
from datasets.rotate_whu_sar_opt_dataset import build_Rotate_WHU 

def build_dataset(args):
    if args.data_name == 'gl':
        train_data_file="/four_disk/image_patch_dataset//stage1/train/train.txt"
        test_data_file="/four_disk/image_patch_dataset//stage1/sar.txt"
        return build_gl(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'RWHU':
        train_data_file="/home/ly/Documents/zkj/dataset/whu-opt-sar/whu_train.txt"
        test_data_file="/home/ly/Documents/zkj/dataset/whu-opt-sar/croped_val/sar.txt"
        return build_Rotate_WHU(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
