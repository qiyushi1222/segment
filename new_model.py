'''测试 把unet嵌入到bi_convgru中
输入的是一系列的图片'''
'''整个网络可视为在序列维度上参数共享的数个二维语义分割网络'''


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
from unet_new import Unet
import torch.nn.functional as F



class Conv2dGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        '''此input_size实际是过了unet的通道数，即64'''
        super(Conv2dGRUCell, self).__init__()

        self.unet = Unet()

        #冻结unet训练参数
        for param in self.unet.parameters():
            param.requires_grad = False

        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.bias = bias
        self.x2h = nn.Conv2d(in_channels=input_size,
                             out_channels=hidden_size * 3,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=hidden_size,
                             out_channels=hidden_size * 3,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.reset_parameters()

        # self.last_conv = nn.Conv2d(hidden_size,2,1)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)
        '''嵌入unet'''
        '''input=(8, 3, 512, 512)经过unet变为(8, 64, 512, 512)'''
        input = self.unet(input)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size, input.size(2), input.size(3)))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        # '''测试都用relu'''
        # reset_gate = torch.sigmoid(x_reset + h_reset)
        # update_gate = torch.sigmoid(x_upd + h_upd)
        # new_gate = F.relu(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate
        # hy = self.last_conv(hy)
        return hy




'''嵌入unet后GRUcell测试'''
# input_size = 64
# hidden_size = 64
# kernel_size = (3, 3)
# model = Conv2dGRUCell(input_size=input_size,
#                       hidden_size=hidden_size,
#                       kernel_size=kernel_size)
# # Inputs:
#         #       input: of shape (batch_size, input_size, height_size, width_size)
#         #       hx: of shape (batch_size, hidden_size, height_size, width_size)
#         # Outputs:
#         #       hy: of shape (batch_size, hidden_size, height_size, width_size)
# batch_size = 8
# height = width = 512
# input_tensor = torch.rand(batch_size, 3, height, width)
# print(input_tensor.size()) #(8, 3, 512, 512)
# out_put = model(input_tensor)
# print(out_put.size()) #(8, 64, 512, 512)



class Conv2dBidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size,
                 kernel_size, num_layers, bias, output_size):
        super(Conv2dBidirRecurrentModel, self).__init__()

        # self.unet = Unet()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if  mode == 'GRU':
            self.rnn_cell_list.append(Conv2dGRUCell(self.input_size,
                                              self.hidden_size,
                                              self.kernel_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dGRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.kernel_size,
                                                  self.bias))


        else:
            raise ValueError("Invalid RNN mode selected.")

        self.conv = nn.Conv2d(in_channels=self.hidden_size * 2,
                             out_channels=self.output_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)
        # self.out_conv = nn.Conv2d(64,2,1)
        self.last_conv = nn.Conv2d(hidden_size * 2,2,1)
    def forward(self, input, hx=None):

        # Input of shape (batch_size, sequence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))

        if torch.cuda.is_available():
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
        else:
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))

        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
            if self.mode == 'LSTM':
                hidden_forward.append((h0[layer], h0[layer]))
            else:
                hidden_forward.append(h0[layer])

        hidden_backward = list()
        for layer in range(self.num_layers):
            if self.mode == 'LSTM':
                hidden_backward.append((hT[layer], hT[layer]))
            else:
                hidden_backward.append(hT[layer])

        for t in range(input.shape[1]):
            for layer in range(self.num_layers):

                if self.mode == 'LSTM':
                    # If LSTM
                    if layer == 0:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            input[:, t, :],
                            (hidden_forward[layer][0], hidden_forward[layer][1])
                            )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            input[:, -(t + 1), :],
                            (hidden_backward[layer][0], hidden_backward[layer][1])
                            )
                    else:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            hidden_forward[layer - 1][0],
                            (hidden_forward[layer][0], hidden_forward[layer][1])
                            )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            hidden_backward[layer - 1][0],
                            (hidden_backward[layer][0], hidden_backward[layer][1])
                            )

                else:
                    # If RNN{_TANH/_RELU} / GRU
                    if layer == 0:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](input[:, t, :], hidden_forward[layer])
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](input[:, -(t + 1), :], hidden_backward[layer])
                    else:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](hidden_forward[layer - 1], hidden_forward[layer])
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](hidden_backward[layer - 1], hidden_backward[layer])


                hidden_forward[layer] = h_forward_l
                hidden_backward[layer] = h_back_l

            if self.mode == 'LSTM':

                outs.append(h_forward_l[0])
                outs_rev.append(h_back_l[0])

            else:
                # h_forward_l = self.last_conv(h_forward_l)
                outs.append(h_forward_l)
                # print(h_forward_l.shape)
                # h_back_l = self.last_conv(h_back_l)
                outs_rev.append(h_back_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)
        # backward是从后往前的，先把顺序调整回来
        outs_rev = outs_rev[::-1]

        '''测试先cat然后卷积'''
        f_b_list = []
        for i in range(len(outs)):
            f_b = torch.cat((outs[i], outs_rev[i]), 1)
            # print(f_b.shape)
            f_b_list.append(f_b)
        # print(len(f_b_list))
        out_f_b_list = []
        for i in range(len(f_b_list)):
            out_f_b = self.last_conv(f_b_list[i])
            # print(out_f_b.shape)
            out_f_b_list.append(out_f_b)
        # print(len(out_f_b_list))
        outs_f_b = torch.cat(out_f_b_list, dim=0)
        # print(outs_f_b.shape)

        '''之前的，不卷积'''
        # 把forward和backward按照seq_len维度进行合并
        # forward = torch.stack(outs, dim=1)
        # backward = torch.stack(outs_rev, dim=1)
        # 把前向和后向按照通道堆叠
        # f_and_b = torch.cat((forward, backward), 2)

        # 把最后的输出通道调整为output_size,一般用不到
        # out = self.conv(out)
        # 返回值一个为最后一层最后一个输出（且通道调整为output_size）
        # 另一个是将前向和后向按照seq_len维度进行压缩后，安装通道堆叠(b, seq_len, c, h, w)

        # 将f_and_b通道调整一下
        # f_and_b = f_and_b.view(-1, 64, input_size,input_size)
        # f_and_b = self.out_conv(f_and_b)
        # f_and_b = f_and_b.view()
        # return out, f_and_b
        return outs_f_b




# inputs = torch.rand(8, 3, 64, 64).cuda()
# print(f'input_size: {inputs.size()} shape: {inputs.device}')
# unet_model = Unet().cuda()
# # print(unet_model(inputs).size())
# features = unet_model(inputs)
# features = features.view(-1, 4, 64, 64, 64)
# print(f'features_size：{features.size()} shape: {features.device}')
#
# mode = 'GRU'
# input_size = features.size(2)
# hidden_size = 31
# kernel_size = (3, 3)
# num_layers = 2
# bias = True
# output_size = 65
#
# model = Conv2dBidirRecurrentModel(mode=mode,
#                                   input_size=input_size,
#                                   hidden_size=hidden_size,
#                                   kernel_size=kernel_size,
#                                   num_layers=num_layers,
#                                   bias=bias,
#                                   output_size=output_size).cuda()
# out, f_and_b = model(features)
# f_and_b = f_and_b.view(-1, 62, 64, 64)
# print(f'outputs_size: {f_and_b.size()} shape: {f_and_b.device}')
'''前向和后向顺序不一样，还得重新看代码'''
'''已修改'''



'''嵌入unet后'''
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
# inputs = torch.rand(2, 3, 1, 320, 320)
# print(f'input_size: {inputs.size()} shape: {inputs.device}')
# mode = 'GRU'
# input_size = 64
# hidden_size = 1
# kernel_size = (3, 3)
# num_layers = 1
# bias = True
# output_size = 65
#
# model = Conv2dBidirRecurrentModel(mode=mode,
#                                   input_size=input_size,
#                                   hidden_size=hidden_size,
#                                   kernel_size=kernel_size,
#                                   num_layers=num_layers,
#                                   bias=bias,
#                                   output_size=output_size)
# out, f_and_b = model(inputs)
# print(f'outputs_size: {f_and_b.size()} shape: {f_and_b.device}')
# print(f'out size: {out.size()}')

from torchsummary import summary

# if __name__ == "__main__":
#     # model = Unet(num_classes = 2, backbone = 'resnet50').train().cuda()
#     model = Conv2dBidirRecurrentModel(mode=mode,
#                                       input_size=input_size,
#                                       hidden_size=hidden_size,
#                                       kernel_size=kernel_size,
#                                       num_layers=num_layers,
#                                       bias=bias,
#                                       output_size=output_size)
#     # print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
#     summary(model, (6, 3, 512, 512))

num_workers = 1

input_size = 64
mode = 'GRU'
hidden_size = 32
kernel_size = (3, 3)
num_layers = 1
bias = True
output_size = 65
model = Conv2dBidirRecurrentModel(mode=mode,
                                      input_size=input_size,
                                      hidden_size=hidden_size,
                                      kernel_size=kernel_size,
                                      num_layers=num_layers,
                                      bias=bias,
                                      output_size=output_size)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
input = torch.rand(2,3,1,320,320)
out = model(input)
print(out.shape) #(6,2,320,320)


if __name__ == "__main__":
    # model = Unet(num_classes = 2, backbone = 'resnet50').train().cuda()
    model = Conv2dBidirRecurrentModel(mode=mode,
                                      input_size=input_size,
                                      hidden_size=hidden_size,
                                      kernel_size=kernel_size,
                                      num_layers=num_layers,
                                      bias=bias,
                                      output_size=output_size)
    # print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    summary(model, (6, 1, 320, 320))