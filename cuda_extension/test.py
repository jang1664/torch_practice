import torch
import myadd

class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return myadd.forward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        grad1, grad2 = myadd.backward(grad_output)
        return grad1, grad2

class AddModule(torch.nn.Module):
    def forward(self, x, y):
        return AddFunction.apply(x, y)

if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True).cuda()
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True).cuda()
    z = AddModule().cuda()(x, y)
    # z.sum().backward()
    # print(x.grad)
    print(z)