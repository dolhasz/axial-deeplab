import torch
import numpy as np
from torchsummary import summary
from lib.models.axialnet import axial50m
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing



class SimpleDecoderBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_ch, in_ch // 2, kernel_size=1) 
        self.bn_1 = torch.nn.BatchNorm2d(in_ch // 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.conv_2 = torch.nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=(1,1))
        self.bn_2 = torch.nn.BatchNorm2d(in_ch // 2)
        self.conv_3 = torch.nn.Conv2d(in_ch // 2, out_ch, kernel_size=1)
        self.bn_3 = torch.nn.BatchNorm2d(out_ch)

        self.conv_m1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1) 
        self.bn_m1 = torch.nn.BatchNorm2d(out_ch)
        self.upsample_m = torch.nn.Upsample(scale_factor=2)

    def forward(self, x, skip=None):
        if isinstance(x, list):
            x, skip = x
        identity = x

        # Main Branch
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)

        # Skip branch
        identity = self.conv_m1(identity)
        identity = self.bn_m1(identity)
        identity = self.relu(identity)
        identity = self.upsample_m(identity)

        x += identity
        if skip is not None:
            x += skip
        x = self.relu(x)

        return x


skips = {}
def get_activation(name):
    def hook(module, input, output):
        skips[name] = output.detach()
    return hook


class AxialDeeplab(torch.nn.Module):
    def __init__(self, backbone, upsampling_block, base_ch=1536):
        super().__init__()
        self.backbone = backbone
        self.backbone[4][2].bn2.register_forward_hook(get_activation(0))
        self.backbone[5][3].bn2.register_forward_hook(get_activation(1))
        self.backbone[6][5].bn2.register_forward_hook(get_activation(2))
        self.up1 = upsampling_block(base_ch, base_ch // 2)
        self.up2 = upsampling_block(base_ch // 2, base_ch // 4)
        self.up3 = upsampling_block(base_ch // 4, base_ch // 8)
        self.up4 = upsampling_block(base_ch // 8, base_ch // 16)
        self.up5 = upsampling_block(base_ch // 16, 3)

    def forward(self, x):
        x = self.backbone(x)
        x = self.up1([x, skips[2]])
        x = self.up2([x, skips[1]])
        x = self.up3([x, skips[0]])
        x = self.up4(x)
        x = self.up5(x)
        return x


def make_deeplab():
    backbone = axial50m(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    print(backbone)
    model = AxialDeeplab(backbone, SimpleDecoderBlock)
    return model



if __name__ == "__main__":
    from lib.datasets.iharmony_2 import iHarmonyLoader
    multiprocessing.set_start_method('spawn')

    # Data
    train_loader = torch.utils.data.DataLoader(
                    iHarmonyLoader('all', train=True), 
                    batch_size=18, shuffle=True, 
                    num_workers=10, pin_memory=True, 
                    drop_last=True)

    val_loader = torch.utils.data.DataLoader(
                    iHarmonyLoader('all', train=False), 
                    batch_size=18, num_workers=10, 
                    pin_memory=True, drop_last=True)

    # Model
    model = make_deeplab()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to('cuda')
    print(model)

    # Optimizer
    lossf = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Tensorboard
    writer = SummaryWriter()

    # Training
    epochs = 100
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        model.train()
        train_loss = 0.0
        for batch, (xb, yb) in enumerate(train_loader):
            print(f'Step {batch}/{len(train_loader)}')
            xb = xb.to('cuda')
            yb = yb.to('cuda')
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
        writer.add_scalar('Loss/Train', train_loss/len(train_loader), epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, (xb, yb) in enumerate(val_loader):
                print(f'Step {idx}/{len(train_loader)}')
                xb = xb.to('cuda')
                yb = yb.to('cuda')
                pred = model(xb)
                loss = lossf(pred, yb)
                val_loss += loss.item()
            writer.add_scalar('Loss/Validation', train_loss/len(val_loader), epoch)
    writer.close()
        