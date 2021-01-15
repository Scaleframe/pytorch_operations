import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32, 
                        device=device, requires_grad=True)


print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# common ways to iniitlaize
x = torch.empty(size = (3,3))
print(x)
x = torch.zeros((3,3))
print(x)

x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5) # identity matrix
x = torch.arange(start=0, end=5, step=1) # 
print(x)

x = torch.linspace(start=0.1, end=1, steps=10) # incremental increase
print(x)

x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # normal distribution
x = torch.empty(size=(1,5)).uniform_(0, 1) # uniform distribution
x = torch.diag(torch.ones(3)) # creates these values at diagonal. 


# how to init and convert tensors to other types, int, float, double
tensor = torch.arange(4)

print(tensor.bool())
