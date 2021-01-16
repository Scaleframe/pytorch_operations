import torch

# print("hello borld")

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
print(tensor.bool()) # boolean True/False
print(tensor.short()) # intzz16
print(tensor.long()) # int64 (Important)
print(tensor.half()) # float16
print(tensor.float()) # float32 (Important)
print(tensor.double()) # float64


# array to tensor conversions
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()



# tensor math operations

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)

# additon operation, all 3 identical
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y

# subtraction
z = x - y

# division:
z = torch.true_divide(x, y) 
# if same shape, will divide element wise
# if not, will broadcast scalar across tensor


# inplace operation
t = torch.zeros(3)
t.add_(x) # can be more computationally efficient. 
t += x # in place
t = t + x # creates a copy

# exponents:
z = x.pow(2)
x = z ** 2 # same operation

# comparison
z = x > 0
z = x < 8