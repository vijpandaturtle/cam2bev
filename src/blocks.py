import torch
from torch import nn
import torchgeometry as tgm
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, inputShape, u_depth, filters1):
        super(Encoder, self).__init__()
        # Define depth
        self.u_depth = u_depth
        # Create a structure for encoder layers 
        self.encoder_layers = nn.ModuleList()
        # Define input filters
        filters = inputShape[-3]
        # output filters 
        filters_ = filters1

        # For each layer implement the block
        for d in range(self.u_depth):
            layer = nn.Sequential( 
                nn.Conv2d(filters, filters_, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters_),
                nn.ReLU(True),
                nn.Conv2d(filters_, filters_, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters_),
                nn.ReLU(True),
            )

            # Add maxpooling to all layers except for the last one
            if d<(self.u_depth-1):
                layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layer.append(nn.Dropout2d(p=0.1))
            
            # Adjust filters to go deeper at the next layer
            filters = filters_
            filters_ = (2**(d+1))*filters1

            self.encoder_layers.append(layer)
    
    def forward(self, x):
        encoder_layers = []
        # Intermediate buffer to modify input to send to the spatial transformer 
        t = x

        for d in range(self.u_depth):
            # Check if it's not the last layer 
            if d<(self.u_depth-1):
                # Apply all but last two operations (maxpool & dropout)
                t = self.encoder_layers[d][:2](t)
                # Store output for spatial transformer units
                encoder_layers.append(t)
                # Now apply last two operations 
                t = self.encoder_layers[d][-2:](t)
            else:
                # The last layer doesn't have Maxpool & Dropout, so we apply all.
                t = self.encoder_layers[d](t)
                encoder_layers.append(t)
        return encoder_layers 

# Spatial Transformer unit is basically uses homography with learnable parameters 
class STN(nn.Module):
    def __init__(self, inputShape, theta):
        super(STN, self).__init__()
        self.theta = theta 
        self.warper = tgm.HomographyWarper(inputShape[-2], inputShape[-1])
    
    def forward(self, x):
        bs = x.shape[0]
        x = self.warper(x, self.theta.repeat(bs, 1, 1))
        return x

# Combining images to create 360 view transform 
class Joiner(nn.Module):
    def __init__(self, inputShape, udepth, n_inputs, filters1, returnWarpedOutput=False):
        super(Joiner, self).__init__()
        self.n_inputs = n_inputs
        self.filters1 = filters1
        self.joiner_outputs = []
        self.joiner_layers = nn.ModuleList()
        self.depth_stn_layers = nn.ModuleList()
        self.returnWarpedOutput = returnWarpedOutput

        H = [
            np.array([[4.651574574230558e-14, 10.192351107009959, -5.36318723862984e-07], [-5.588661045867985e-07, 0.0, 2.3708767903941617], [35.30731833118676, 0.0, -1.7000018578614013]]),                                       # front
            np.array([[20.38470221401992, 7.562206982469407e-14, -0.28867638384075833], [-3.422067857504854e-23, 2.794330463189411e-07, 2.540225111648729], [2.1619497190382224e-15, -17.65365916559334, -0.4999990710692976]]),    # left
            np.array([[-5.336674306912119e-14, -10.192351107009957, 5.363187220578325e-07], [5.588660952931949e-07, 3.582264351370481e-23, 2.370876772982613], [-35.30731833118661, -2.263156574813233e-15, -0.5999981421386035]]), # rear
            np.array([[-20.38470221401991, -4.849709834037436e-15, 0.2886763838407495], [-3.4220679184765114e-23, -2.794330512976549e-07, 2.5402251116487626], [2.161949719038217e-15, 17.653659165593304, -0.5000009289306967]])   # right
            ]

        H = [torch.from_numpy(x).float().to(device) for x in H]

        shape = list(inputShape)
        for d in range(udepth):
            layer = []
            filters = (2**d)*self.filters1
            warped_maps = []
            stn_layers = nn.ModuleList()
            for i in range(self.n_inputs):
                shape[-3] = filters
                shape[-2] = inputShape[-2]//(2**d)
                shape[-1] = inputShape[-1]//(2**d)
                stn_layers.append(STN(tuple(shape), H[i]))
            
            self.depth_stn_layers.append(stn_layers)
            layer.append(nn.Conv2d(filters*n_inputs, filters, kernel_size=3, padding = 1))
            layer.append(nn.BatchNorm2d(filters))
            layer.append(nn.ReLU(True))
            layer.append(nn.Conv2d(filters, filters, kernel_size=3, padding = 1))
            layer.append(nn.BatchNorm2d(filters))
            layer.append(nn.ReLU(True))
            self.joiner_layers.append(nn.Sequential(*layer))
        
    def warpedOutput(self, flag):
        self.returnWarpedOutput = flag 
    
    def forward(self, encoder_outputs):
        self.joiner_outputs = []
        warped_outputs = []

        for d in range(len(encoder_outputs[0])):
            filters = (2**d)*self.filters1
            warped_maps = []
            for i in range(self.n_inputs):
                t = self.depth_stn_layers[d][i](encoder_inputs[i][d])
                warped_maps.append(t)
            t = torch.cat(warped_maps, dim=1) if self.n_inputs > 1  else warped_maps[0]
            if(self.returnWarpedOutput):
                warped_outputs.append([x.cpu().detach().numpy() for x in warped_maps])
            t = self.joiner_layers[d](t)
            self.joiner_outputs.append(t)

        if (self.returnWarpedOutput):
            return self.joiner_outputs, warped_outputs
        else: 
            return self.joiner_outputs

# Decoder module to upsample to transformed image  
class Decoder(nn.Module):
    def __init__(self, udepth, filters1):
        super(Decoder, self).__init__()
        self.udepth = udepth
        self.filters1 = filters1
        self.decoder_layers = nn.ModuleList()
        
        for d in reversed(range(self.udepth-1)):
            filters = (2**d)*self.filters1
            layer = []
            layer.append(nn.ConvTranspose2d(filters*2, filters, kernel_size=3, stride=2, padding=1, output_padding=1))
            layer.append(nn.Dropout2d(p=0.1))
            layer.append(nn.Conv2d(filters*2, filters, kernel_size=3, padding=1))
            layer.append(nn.BatchNorm2d(filters))            
            layer.append(nn.ReLU(True))
            layer.append(nn.Conv2d(filters, filters, kernel_size=3, padding=1))
            layer.append(nn.BatchNorm2d(filters))            
            layer.append(nn.ReLU(True))
            self.decoder_layers.append(nn.Sequential(*layer))

    def forward(self, joiner_outputs):
        t = joiner_outputs[-1]
        for d in reversed(range(self.udepth-1)):
            filters = (2**d)*self.filters1
            t = self.decoder_layers[self.udepth-2-d][0](t)
            t = torch.cat((joiner_outputs[d], t), dim=1)
            t = self.decoder_layers[self.udepth-2-d][1:](t)
        return t


# Test
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoderModule = Encoder(inputShape=(10,256,512), u_depth=4, filters1=16).to(device)

# joinerModule = Joiner(inputShape=(10,256,512), udepth=4, n_inputs=4, filters1=16, returnWarpedOutput=True).to(device)
# joinerOutput, warped_outputs = joinerModule([encoderOutput for i in range(4)])   # simulate inputs from 4 cameras
# print(f"Joiner module output = {[x.shape for x in joinerOutput]}")

