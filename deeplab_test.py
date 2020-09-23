import torch
import numpy as np
from torchsummary import summary
from lib.models.axialnet import axial50m
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing
from lib.datasets.iharmony_2 import iHarmonyLoader
from lib.models.axialnet import AxialDecoderBlock


class SimpleDecoderBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, conv_skip=False):
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

        if conv_skip:
            self.conv_skip = torch.nn.Conv2d(48, 96, kernel_size=1)

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
            if hasattr(self, 'conv_skip'):
                skip = self.conv_skip(skip)
            x += skip
        x = self.relu(x)

        return x


class AxialDeeplab(torch.nn.Module):
    def __init__(self, backbone, upsampling_block, base_ch=1536):
        super().__init__()
        self.backbone = backbone
        self.backbone[1].register_forward_hook(get_activation(0))
        self.backbone[4][2].bn2.register_forward_hook(get_activation(1))
        self.backbone[5][3].bn2.register_forward_hook(get_activation(2))
        self.backbone[6][5].bn2.register_forward_hook(get_activation(3))
        
        # self.up1 = upsampling_block(base_ch, base_ch // 2)

        self.up1 = AxialDecoderBlock(base_ch, base_ch // 2, norm_layer=torch.nn.BatchNorm2d, kernel_size=14, groups=8, skip=True)
        self.up2 = AxialDecoderBlock(base_ch // 2, base_ch // 4, norm_layer=torch.nn.BatchNorm2d, kernel_size=28, groups=8, skip=True)
        self.up3 = AxialDecoderBlock(base_ch // 4, base_ch // 8, norm_layer=torch.nn.BatchNorm2d, kernel_size=56, groups=8, skip=True)
        self.up4 = AxialDecoderBlock(base_ch // 8, base_ch // 16, norm_layer=torch.nn.BatchNorm2d, kernel_size=112, groups=8, skip=True, hack=True ) # FIXME: This is hacky - find out what's happening
        # self.up5 = AxialDecoderBlock(base_ch // 16, 3, norm_layer=torch.nn.BatchNorm2d, kernel_size=224, skip=False)
        # self.up2 = upsampling_block(base_ch // 2, base_ch // 4)
        # self.up3 = upsampling_block(base_ch // 4, base_ch // 8)
        # self.up4 = upsampling_block(base_ch // 8, base_ch // 16, conv_skip=True) # FIXME: This is hacky - find out what's happening
        self.up5 = upsampling_block(base_ch // 16, 3)

    def forward(self, x):
        x = self.backbone(x)
        x = self.up1([x, skips[3].to('cuda')])
        x = self.up2([x, skips[2].to('cuda')])
        x = self.up3([x, skips[1].to('cuda')])
        x = self.up4([x, skips[0].to('cuda')])
        x = self.up5(x)
        return x


skips = [None for _ in range(4)]
def get_activation(name):
    def hook(module, input, output):
        skips[name] = output.detach().to('cpu')
    return hook


def make_deeplab():
    backbone = axial50m(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    print(backbone)
    model = AxialDeeplab(backbone, SimpleDecoderBlock)
    return model


def run_epoch(model, dataloader, lossf, opt=None):
    epoch_loss = 0.0
    for batch, (xb, yb) in enumerate(dataloader):
        print(f'Step {batch}/{len(dataloader)}')
        if opt is None:
            batch_loss, n, pred = loss_batch(model, lossf, xb, yb, opt)
        else:
            batch_loss, n = loss_batch(model, lossf, xb, yb, opt)
        epoch_loss += batch_loss
    if opt is None:
        return epoch_loss, n, pred
    else:
        return epoch_loss, n


def loss_batch(model, lossf, xb, yb, opt=None):
    xb = xb.to('cuda')
    yb = yb.to('cuda')
    pred = model(xb)
    loss = lossf(pred, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    if opt is None:
        return loss.item(), len(xb), pred
    else:
        return loss.item(), len(xb)


def fit(model, train_loader, val_loader, lossf, opt, epochs=100):
    # Tensorboard
    writer = SummaryWriter()
    print(f'LOG DIR IS: {writer.log_dir}')

    # Training
    lowest_loss = 100.0
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')

        model.train()
        train_loss, n = run_epoch(model, train_loader, lossf, opt)
        writer.add_scalar('Loss/Train', train_loss/len(train_loader), epoch)

        model.eval()
        with torch.no_grad():
            vloss, n, pred = run_epoch(model, val_loader, lossf, opt=None)
            epoch_vloss = vloss/len(val_loader)
            writer.add_scalar('Loss/Validation', epoch_vloss, epoch)
            for p in pred:
                writer.add_image('img', p.reshape(3, 224, 224), epoch)
            if epoch_vloss < lowest_loss:
                lowest_loss = epoch_vloss
                torch.save(model.state_dict(), f'{writer.get_logdir()}/best_model.pt')
                
    writer.close()



if __name__ == "__main__":
    
    # Params
    epochs = 100
    batch_size = 4
    lr = 0.001
    dataset = 'Hday2night'

    n_gpus = torch.cuda.device_count()
    n_cpus = multiprocessing.cpu_count()
    multiprocessing.set_start_method('spawn')

    # Data
    train_loader = torch.utils.data.DataLoader(
                    iHarmonyLoader(dataset, train=True), 
                    batch_size=batch_size*n_gpus, shuffle=True, 
                    num_workers=n_cpus, pin_memory=True, 
                    drop_last=True)

    val_loader = torch.utils.data.DataLoader(
                    iHarmonyLoader(dataset, train=False), 
                    batch_size=batch_size*n_gpus, num_workers=n_cpus, 
                    pin_memory=True, drop_last=True)

    # Model
    model = make_deeplab()
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.to('cuda')
    print(model)

    # Optimizer
    lossf = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    fit(model, train_loader, val_loader, lossf, opt, epochs)
        