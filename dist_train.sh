python main_moco_mv.py \
  -a resnet50 \
  --lr 0.03 \
  --pretrained ./checkpoints/moco_v2_200ep_pretrain.pth.tar \
  --batch-size 256 \
  --mlp \
  --moco-t 0.2 \
  --aug-plus \
  --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  ./data
