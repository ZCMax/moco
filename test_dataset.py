from moco.dataset import MVImageDataset
import torchvision.transforms as transforms
import moco.loader
import os

traindir = './data/train/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
]

train_dataset = MVImageDataset(
    traindir,
    moco.loader.MVTwoCropsTransform(transforms.Compose(augmentation)))

files = os.listdir(traindir)

for file in files:
    f=str(traindir+file)    #使用绝对路径
    if not os.listdir(f):
        print(f)
print(len(os.listdir(traindir)))
print(len(train_dataset))


# for i in range(1):
#     sample = train_dataset[i]
#     print(len(sample))
#     sample[0].save('sample_0.jpg')
#     sample[1].save('sample_1.jpg')
# print(sample[0])

