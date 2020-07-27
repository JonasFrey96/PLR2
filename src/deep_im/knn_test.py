import torch
from torch_cluster import nearest
import time
# x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])

# batch_x = torch.tensor([0, 0, 0, 0])


# y = torch.Tensor([[-1, 0], [1, 0]])

# batch_y = torch.tensor([0, 0])

bs = 1
nr_points = 3000
nr_points_tar = nr_points

model_points = torch.rand((nr_points, 3), device='cuda:0')
idx_mp = torch.arange(0, nr_points).cuda()

rand_index = torch.randperm(nr_points).cuda()
target_points = model_points[rand_index, :].cuda()
idx_tp = torch.arange(0, nr_points).cuda()

mp2 = torch.zeros((nr_points_tar, nr_points, 3))
tp2 = torch.zeros((nr_points_tar, nr_points, 3))

test = torch.randn(1, 3)

start = time.time()

mp2 = model_points.unsqueeze(0).repeat(nr_points_tar, 1, 1)
tp2 = target_points.unsqueeze(1).repeat(1, nr_points, 1)
# for i in range(0, nr_points):
#     tp2[i, :, :] = tp2[i, 0, :].unsqueeze(0).repeat(nr_points, 1)
dist = torch.norm(mp2 - tp2, dim=2, p=None)
knn = dist.topk(1, largest=False)

print(f'Took {time.time()-start}s')
