import torch

#DT和IDT内部只进行水平方向（从左到右）处理，如果想进行逆水平、垂直、逆垂直方向处理，可以在输入前，对x和x_DT分别进行转置或翻转操作即可

def DT(self, x):

    b, c, h, w = x.shape  # 读取输入张量形状：batch、channel、高、宽

    dIcdx = (x[:, :, :, 1:] - x[:, :, :, :-1])  # 沿宽度方向做一阶差分：当前列减前一列

    dIdx = torch.zeros(b, c, h, w).to(x.device)  # 初始化差分幅值张量（与输入同形状），放到同一设备
    kx = torch.zeros(b, c, h, w).to(x.device)  # 初始化差分符号张量（与输入同形状），放到同一设备

    dIdx[:, :, :, 1:] = torch.abs(dIcdx[:, :, :, :])  # 从第2列开始填入差分绝对值（第1列保留0）
    kx[:, :, :, 1:] = dIcdx[:, :, :, :]  # 从第2列开始填入原始差分（带正负号）

    dHdx = 1 + dIdx  # 将差分幅值整体平移 +1（保证值至少为1）

    kx = torch.where(kx >= 0.0, torch.tensor(1.0), torch.tensor(-1.0))  # 将差分符号二值化：非负为+1，负为-1

    x_DT = dHdx.view(b, -1, h * w)  # 将 dHdx 从 (b,c,h,w) 拉平成 (b,c,h*w)
    k = kx.view(b, -1, h * w)  # 将 kx 从 (b,c,h,w) 拉平成 (b,c,h*w)

    return x_DT, k  # 返回“幅值编码结果 x_DT”和“符号编码结果 k”

def IDT(self, x, x_DT, k):

    b, c, h, w = x.shape  # 读取输入张量形状：batch、channel、高、宽

    x_DT = x_DT.view(b, -1, h, w)  # 还原为 (b,c,h,w)
    k = k.view(b, -1, h, w)        # 还原为 (b,c,h,w)

    abs_diff = x_DT - 1  # 恢复绝对差分幅值

    diff = abs_diff * k # 用符号恢复真实差分（可正可负）

    acc = torch.cumsum(diff, dim=3)  # 沿宽度方向积分（前缀和）

    x0 = x[:, :, :, 0].unsqueeze(3).expand(b, c, h, w)  # 用原始首列作为边界条件重建

    x_H = x0 + acc  #用输入 x 的首列值作为基准 x[...,0]，加上累计增量得到 x_H

    return x_H  # 返回保边滤波结果 x_H