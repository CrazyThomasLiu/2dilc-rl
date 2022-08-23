import pdb
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = torch.zeros(3, 3, device=device)
print(t.device)


t = torch.zeros(3, 3)
print(t.device)

t = t.to(device)
print(t.device)


"只有一个GPU时，我们还可以对张量调用 cuda() 方法来返回一个在GPU上的拷贝"
t = torch.zeros(3, 3)
t = t.cuda()
print(t.device)