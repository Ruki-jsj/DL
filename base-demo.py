import torch
x=torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)
y=x+2
print(y)
print(y.grad_fn)
#x为直接创建，没有grad_fn，x被称作叶子节点，对应的为none
print(x.is_leaf,y.is_leaf)
z=y*y*3
out=z.mean()#mean函数求取均值
print(z,out)

a = torch.randn(2, 2) #requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)
out.backward()
print(x.grad)



out2 = x.sum()
out2.backward()
print(x.grad)
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

x = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y = 2*x
z = y.view(2,2)
print(z)
v=torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
z.backward(v)
print(x.grad)

x=torch.tensor(1.0,requires_grad=True)
y1=x**2
with torch.no_grad(): # 里边所有相关梯度不会回传
    y2=x**3
y3=y1+y2
print(x.requires_grad)
print(y1,y1.requires_grad)
print(y2,y2.requires_grad)
print(y3,y3.requires_grad)
y3.backward()
print(x.grad)

x = torch.ones(1,requires_grad=True)
print(x.data)
print(x.data.requires_grad)  # 独立于计算图之外

y=2*x
x.data*= 100  # 只改变了值，但不会记录在计算图中，不会影响梯度传播

y.backward()
print(x) # 只是更改值
print(x.grad)
