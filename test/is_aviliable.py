import torch
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

#pdb.set_trace()
#a=2