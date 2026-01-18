epochs: 50
batch: 8
lr: 0.0002
data: data/train/good

loss_weights:
  gan: 1.0
  l1: 10.0
  ssim: 2.0
  freq: 1.0
  lpips: 1.0
