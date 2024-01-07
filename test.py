from dataset import Model10NetDataset

dataset = Model10NetDataset(path="./ModelNet10", train=True)

print(len(dataset))

for x in dataset:
    print(x)
    break
