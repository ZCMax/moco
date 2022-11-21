import mmengine

# MVI文件夹路径

pth="./data/val"

# 写入文件夹路径
wt_pth = "./cls_name_lst_val"
mmengine.mkdir_or_exist(wt_pth)

from pathlib import Path 
pth = Path(pth)
wt_pth = Path(wt_pth)
import os 

counter = 0 
for cls_pth in pth.iterdir():
    if len(cls_pth.name) > 4 :
        continue 

    dest_cls_file = wt_pth / Path(cls_pth.name + ".txt")
    if not os.path.exists(str(dest_cls_file)):
        dest_cls_file = open(str(dest_cls_file) , 'w')
    else:
        dest_cls_file = open(str(dest_cls_file) , 'a')

    for vd_pth in cls_pth.iterdir():
        vdname = vd_pth.name
        dest_cls_file.write(vdname + "\n")
        counter += 1 


print(counter)