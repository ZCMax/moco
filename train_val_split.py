import os,random,shutil
from moco.dataset import find_mv_instances
import mmengine 

def moveFile(fileDir):
    instance_dirs = find_mv_instances(fileDir)
    for instance_dir in mmengine.track_iter_progress(instance_dirs):
        pathDir = os.listdir(instance_dir)  # 取图片的原始路径
        if len(pathDir) > 5:
            sample = random.sample(pathDir, 5)  # 随机选取5张样本图片
        else:
            sample = pathDir
        tar_dir = instance_dir.replace('MVImgNet_v08', 'MVImgNet_v08_train')
        mmengine.mkdir_or_exist(tar_dir)
        for name in sample:
            instance_file = os.path.join(instance_dir, name)
            tar_file = os.path.join(tar_dir, name)
            shutil.move(instance_file, tar_file)

def movebackFile(fileDir):
    instance_dirs = find_mv_instances(fileDir)
    for instance_dir in instance_dirs:
        sample = os.listdir(instance_dir)  # 取图片的原始路径
        # sample = random.sample(pathDir, 5)  # 随机选取5张样本图片
        tar_dir = instance_dir.replace('MVImgNet_v08_train', 'MVImgNet_v08')
        # mmengine.mkdir_or_exist(tar_dir)
        for name in sample:
            instance_file = os.path.join(instance_dir, name)
            tar_file = os.path.join(tar_dir, name)
            shutil.move(instance_file, tar_file)
 
if __name__ == '__main__':
    fileDir = "/root/data/MVImgNet_v08/"  # 源图片文件夹路径
    # tarDir = '/home/xiaobumidm/Yolo_mark-master/验证/验证集/明显/'  # 移动到新的文件夹路径
    moveFile(fileDir)

