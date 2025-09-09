import torch

# File paths
pt1_path = 'subset1_40x40.pt'
pt2_path = 'subset2_40x40.pt'
output_path = 'no-overlap_40x40.pt'

# Load tensors
data1 = torch.load(pt1_path)
data2 = torch.load(pt2_path)

print('data1: ', data1.shape)
print('data2: ', data2.shape)

len1 = data1.size(0)
len2 = data2.size(0)

n_remove = len2 - len1
if n_remove > 0:
    print(f"Removing {n_remove} images from the second tensor for balance.")
    data2 = data2[:len2 - n_remove]

print('data1: ', data1.shape)
print('data2: ', data2.shape)

# Concatenate along the batch dimension (0)
merged = torch.cat((data1, data2), dim=0)

print('merged: ', merged.shape)

# Save the merged tensor
torch.save(merged, output_path)
print(f"Merge complete. Merged tensor size: {merged.size()}")
