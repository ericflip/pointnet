# pointnet

Implementation of PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

https://arxiv.org/pdf/1612.00593.pdf

### Training Script

```bash
python3 train.py \
	--dataset-path=./ModelNet10 \
	--batch=32 \
	--lr=1e-3 \
	--epochs=100 \
	--checkpoint-every=1 \
	--evaluate-every=1 \
	--checkpoint-path=./checkpoint-big-100 \
	--jitter-augmentation=True \
    --rotation-augmentation=False
```

### Inference Script

```bash
python inference.py \
	--checkpoint-path=./checkpoint-big-100/epoch-96.pt \
	--obj-path=./ModelNet10/bathtub/train/bathtub_0001.off
```

### Training

We trained PointNet on ModelNet10 for 100 epochs with a batch size of 32 and a learning rate of 1e-3 using the Adam optimizer. We sample 1024 points from the surface of the mesh, normalize them to an unit sphere, and add gaussian noise with a mean of 0 and standard deviation of 0.02.

### Classification Accuracy (Overall + Per Class)

| Overall | Bathtub | Bed   | Chair | Desk  | Dresser | Monitor | Nightstand | Sofa   | Table | Toilet |
| ------- | ------- | ----- | ----- | ----- | ------- | ------- | ---------- | ------ | ----- | ------ |
| 90.2%   | 92.0%   | 95.0% | 98.0% | 88.4% | 91.9%   | 96.0%   | 55.8%      | 100.0% | 85.0% | 96.0%  |
