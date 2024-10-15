from blocks import Encoder, STN, Joiner, Decoder 

class UNetXST(nn.Module):
    def __init__(self, nInputs, nOutputClasses, udepth, filters1, returnWarpedOutput=False):
        super(UNetXST, self).__init__()
        self.inputShape = inputShape
        self.nInputs = nInputs
        self.nOutputClasses = nOutputClasses
        self.udepth = udepth
        self.filters1 = filters1
        self.returnWarpedOutput = returnWarpedOutput

        self.encoder = nn.ModuleList([Encoder(self.inputShape, self.udepth, self.filters1) for i in range(nInputs)])
        self.joiner = Joiner(self.inputShape, self.udepth, self.nInputs, self.filters1, returnWarpedOutput=self.returnWarpedOutput)
        self.decoder = Decoder(self.udepth, self.filters1)

        self.prediction = nn.Sequential(
            nn.Conv2d(self.filters1, nOutputClasses, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, inputs):
        encoder_outputs = []
        for i in range(self.nInputs):
            encoder_outputs.append(self.encoder[i](inputs[i]))

        if self.returnWarpedOutput:
            joiner_output, warped_outputs = self.joiner(encoder_outputs)
            decoder_output = self.decoder(joiner_output)
            prediction = self.prediction(decoder_output)
            return prediction, warped_outputs

        else:
            joiner_output = self.joiner(encoder_outputs)
            decoder_output = self.decoder(joiner_output)
            prediction = self.prediction(decoder_output)
            return prediction

#Test
#model = UNetXST(inputShape =(10, 256, 512), nInputs= 4, nOutputClasses=10, udepth=4, filters1=16, returnWarpedOutput=True).to(device)


