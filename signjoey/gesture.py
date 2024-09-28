a = torch.arange(10).reshape(5,2)
torch.split(a, 2)
torch.split(a, [1,4])