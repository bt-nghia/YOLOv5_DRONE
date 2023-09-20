# main model
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode


# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     # Pad to 'same' shape outputs
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# performs a convolution, a batch_norm and then applies a SiLU activation function
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, g=1, d=1):
        super(Conv, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        # bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.conv = nn.Sequential(
            conv,
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # print(self.Conv(x).shape)
        return self.conv(x)


# which is just a residual block
class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(Bottleneck, self).__init__()
        c_ = int(width_multiple * in_channels)
        self.c1 = Conv(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = Conv(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x


class C3(nn.Module):

    def __init__(self, in_channels, out_channels, width_multiple=0.5, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple * out_channels)

        self.c1 = Conv(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = Conv(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_out = Conv(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    Conv(c_, c_, 1, 1, 0),
                    Conv(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels // 2)

        self.c1 = Conv(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = Conv(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))


class C3_NECK(nn.Module):
    def __init__(self, in_channels, out_channels, width, depth):
        super(C3_NECK, self).__init__()
        c_ = int(in_channels * width)
        self.in_channels = in_channels
        self.c_ = c_
        self.out_channels = out_channels
        self.c_skipped = Conv(in_channels, c_, 1, 1, 0)
        self.c_out = Conv(c_ * 2, out_channels, 1, 1, 0)
        self.silu_block = self.make_silu_block(depth)

    def make_silu_block(self, depth):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(Conv(self.in_channels, self.c_, 1, 1, 0))
            elif i % 2 == 0:
                layers.append(Conv(self.c_, self.c_, 3, 1, 1))
            elif i % 2 != 0:
                layers.append(Conv(self.c_, self.c_, 1, 1, 0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.c_out(torch.cat([self.silu_block(x), self.c_skipped(x)], dim=1))


class HEADS(nn.Module):
    def __init__(self, nc=1, anchors=(), ch=()):  # detection layer
        super(HEADS, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = 3
        self.na = 3
        self.stride = [8, 16, 32]
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6,
                                                                                                         1).T.reshape(3,
                                                                                                                      3,
                                                                                                                      2)
        self.register_buffer('anchors', anchors_)  # shape(nl,na,2)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(5 + self.nc) * 3, kernel_size=1)
            ]

    def forward(self, x):

        z = []  # inference information
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs, _, ny, nx = x[i].shape


            # performs out_convolution and stores the result in place

            # bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5 + self.nc), ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, 6))
        return (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=torch.__version__):
        # d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, dtype=t), torch.arange(nx, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class YOLOV5n(nn.Module):
    def __init__(self, first_out, nc=1, anchors=(),
                 ch=(), inference=False):
        super(YOLOV5n, self).__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [
            Conv(in_channels=3, out_channels=16, kernel_size=6, stride=2, padding=2),  # 3 16 6 2 2
            Conv(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 16 32 3 2 1
            C3(in_channels=32, out_channels=32, width_multiple=0.5, depth=1),  # 32, 32
            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            C3(in_channels=64, out_channels=64, width_multiple=0.5, depth=2),
            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            C3(in_channels=128, out_channels=128, width_multiple=0.5, depth=3),
            Conv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            C3(in_channels=256, out_channels=256, width_multiple=0.5, depth=1),
            SPPF(in_channels=256, out_channels=256)
        ]

        self.neck = nn.ModuleList()
        self.neck += [
            Conv(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            C3(in_channels=256, out_channels=128, width_multiple=0.5, depth=1, backbone=False),
            Conv(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            C3(in_channels=128, out_channels=64, width_multiple=0.5, depth=1, backbone=False),
            Conv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            C3(in_channels=128, out_channels=128, width_multiple=0.5, depth=1, backbone=False),
            Conv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            C3(in_channels=256, out_channels=256, width_multiple=0.5, depth=1, backbone=False)
        ]
        self.head = HEADS(nc=nc, anchors=anchors, ch=ch)

    def forward(self, x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        backbone_connection = []
        neck_connection = []
        outputs = []
        for idx, layer in enumerate(self.backbone):
            # takes the out of the 2nd and 3rd C3 block and stores it
            x = layer(x)
            if idx in [4, 6]:
                backbone_connection.append(x)

        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = Resize([x.shape[2] * 2, x.shape[3] * 2], interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                # print(idx)
                # print(layer)
                x = layer(x)
                # print(x.shape)
                # print(neck_connection.pop(-1).shape)

                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            elif (isinstance(layer, C3_NECK) and idx > 2) or (isinstance(layer, C3) and idx > 2):
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)

        return self.head(outputs)


if __name__ == "__main__":
    batch_size = 2
    image_height = 640
    image_width = 640
    nc = 1
    anchors = ([[10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326]]) # P5/32
    x = torch.rand(batch_size, 3, image_height, image_width)
    first_out = 0

    model = YOLOV5n(first_out=first_out, nc=nc, anchors=anchors,
                    ch=([64, 128, 256]), inference=False)
    print(model)
    # out = model(x)