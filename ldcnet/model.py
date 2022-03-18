from ldcnet.basic import *
from ldcnet.CoordConv import AddCoordsNp
import numpy as np
import torch

class ENet(nn.Module):
    def __init__(self, h, w):
        super(ENet, self).__init__()
        self.geoplanes = 3
        self.geofeature = GeometryFeature()
        self.h = h
        self.w = w

        # rgb encoder
        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)


        # depth encoder
        self.depth_conv_init = convbnrelu(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.depth_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.depth_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.depth_layer3 = BasicBlockGeo(inplanes=128, planes=128, stride=2, geoplanes=self.geoplanes)
        self.depth_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.depth_layer5 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.depth_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.depth_layer7 = BasicBlockGeo(inplanes=512, planes=512, stride=2, geoplanes=self.geoplanes)
        self.depth_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.depth_layer9 = BasicBlockGeo(inplanes=1024, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.depth_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        # decoder
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.decoder_layer6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, input, args):
        rgb = input[:,0:3,:,:]
        d = input[:,3:4,:,:]

        position = args["position"]
        K = args["K"]

        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]


        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None

        geo_s1 = self.geofeature(d, vnorm, unorm, self.h, self.w, c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, self.h / 2, self.w / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, self.h / 4, self.w / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, self.h / 8, self.w / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, self.h / 16, self.w / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, self.h / 32, self.w / 32, c352, c1216, f352, f1216)

        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2)
        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2)
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3)
        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3)
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4)
        rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4)
        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5)
        rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5)
        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6)
        rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6)

        rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
        rgb_feature8_plus = rgb_feature_decoder8 + rgb_feature8

        rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
        rgb_feature6_plus = rgb_feature_decoder6 + rgb_feature6

        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
        rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4

        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2

        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature

        rgb_output = self.rgb_decoder_output(rgb_feature0_plus)
        rgb_depth = rgb_output[:, 0:1, :, :]
        rgb_conf = rgb_output[:, 1:2, :, :]

        sparsed_feature = self.depth_conv_init(torch.cat((d, rgb_depth), dim=1))
        sparsed_feature1 = self.depth_layer1(sparsed_feature, geo_s1, geo_s2)
        sparsed_feature2 = self.depth_layer2(sparsed_feature1, geo_s2, geo_s2)

        sparsed_feature2_plus = torch.cat([rgb_feature2_plus, sparsed_feature2], 1)
        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus, geo_s2, geo_s3)
        sparsed_feature4 = self.depth_layer4(sparsed_feature3, geo_s3, geo_s3)

        sparsed_feature4_plus = torch.cat([rgb_feature4_plus, sparsed_feature4], 1)
        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus, geo_s3, geo_s4)
        sparsed_feature6 = self.depth_layer6(sparsed_feature5, geo_s4, geo_s4)

        sparsed_feature6_plus = torch.cat([rgb_feature6_plus, sparsed_feature6], 1)
        sparsed_feature7 = self.depth_layer7(sparsed_feature6_plus, geo_s4, geo_s5)
        sparsed_feature8 = self.depth_layer8(sparsed_feature7, geo_s5, geo_s5)

        sparsed_feature8_plus = torch.cat([rgb_feature8_plus, sparsed_feature8], 1)
        sparsed_feature9 = self.depth_layer9(sparsed_feature8_plus, geo_s5, geo_s6)
        sparsed_feature10 = self.depth_layer10(sparsed_feature9, geo_s6, geo_s6)

        # -----------------------------------------------------------------------------------------

        fusion1 = rgb_feature10 + sparsed_feature10
        decoder_feature1 = self.decoder_layer1(fusion1)

        fusion2 = sparsed_feature8 + decoder_feature1
        decoder_feature2 = self.decoder_layer2(fusion2)

        fusion3 = sparsed_feature6 + decoder_feature2
        decoder_feature3 = self.decoder_layer3(fusion3)

        fusion4 = sparsed_feature4 + decoder_feature3
        decoder_feature4 = self.decoder_layer4(fusion4)

        fusion5 = sparsed_feature2 + decoder_feature4
        decoder_feature5 = self.decoder_layer5(fusion5)

        depth_output = self.decoder_layer6(decoder_feature5)
        d_depth, d_conf = torch.chunk(depth_output, 2, dim=1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf, d_conf), dim=1)), 2, dim=1)
        output = rgb_conf*rgb_depth + d_conf*d_depth

        return rgb_depth, d_depth, output


class LDCNet(nn.Module):
    def __init__(self, h, w):
        super(LDCNet, self).__init__()

        self.geoplanes = 3
        self.geofeature = GeometryFeature()        
        self.h = h
        self.w = w

        # rgb encoder
        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, input, args):
        rgb = input[:,0:3,:,:]
        d = input[:,3:4,:,:]

        position = args["position"]
        K = args["K"]

        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
        
        geo_s1 = self.geofeature(d, vnorm, unorm, self.h, self.w, c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, self.h / 2, self.w / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, self.h / 4, self.w / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, self.h / 8, self.w / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, self.h / 16, self.w / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, self.h / 32, self.w / 32, c352, c1216, f352, f1216)

        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2)
        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2)
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3)
        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3)
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4)
        rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4)
        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5)
        rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5)
        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6)
        rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6)

        rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
        rgb_feature8_plus = rgb_feature_decoder8 + rgb_feature8

        rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
        rgb_feature6_plus = rgb_feature_decoder6 + rgb_feature6

        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
        rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4

        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2

        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature

        rgb_output = self.rgb_decoder_output(rgb_feature0_plus)
        rgb_depth = rgb_output[:, 0:1, :, :]

        return rgb_depth

# class LDCNet(nn.Module):
#     def __init__(self, h, w):
#         super(LDCNet, self).__init__()

#         self.geoplanes = 3
#         self.geofeature = GeometryFeature()        
#         self.h = h
#         self.w = w

#         # rgb encoder
#         self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)

#         self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
#         self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

#         self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.rgb_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.rgb_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

#         self.decoder_layer6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
#         self.softmax = nn.Softmax(dim=1)
#         self.pooling = nn.AvgPool2d(kernel_size=2)
#         self.sparsepooling = SparseDownSampleClose(stride=2)

#         weights_init(self)

#     def forward(self, input, args):
#         rgb = input[:,0:3,:,:]
#         d = input[:,3:4,:,:]

#         position = args["position"]
#         K = args["K"]

#         unorm = position[:, 0:1, :, :]
#         vnorm = position[:, 1:2, :, :]

#         f352 = K[:, 1, 1]
#         f352 = f352.unsqueeze(1)
#         f352 = f352.unsqueeze(2)
#         f352 = f352.unsqueeze(3)
#         c352 = K[:, 1, 2]
#         c352 = c352.unsqueeze(1)
#         c352 = c352.unsqueeze(2)
#         c352 = c352.unsqueeze(3)
#         f1216 = K[:, 0, 0]
#         f1216 = f1216.unsqueeze(1)
#         f1216 = f1216.unsqueeze(2)
#         f1216 = f1216.unsqueeze(3)
#         c1216 = K[:, 0, 2]
#         c1216 = c1216.unsqueeze(1)
#         c1216 = c1216.unsqueeze(2)
#         c1216 = c1216.unsqueeze(3)

#         vnorm_s2 = self.pooling(vnorm)
#         vnorm_s3 = self.pooling(vnorm_s2)
#         vnorm_s4 = self.pooling(vnorm_s3)
#         vnorm_s5 = self.pooling(vnorm_s4)
#         vnorm_s6 = self.pooling(vnorm_s5)

#         unorm_s2 = self.pooling(unorm)
#         unorm_s3 = self.pooling(unorm_s2)
#         unorm_s4 = self.pooling(unorm_s3)
#         unorm_s5 = self.pooling(unorm_s4)
#         unorm_s6 = self.pooling(unorm_s5)

#         valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
#         d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
#         d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
#         d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
#         d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
#         d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

#         geo_s1 = None
#         geo_s2 = None
#         geo_s3 = None
#         geo_s4 = None
#         geo_s5 = None
#         geo_s6 = None
        
#         geo_s1 = self.geofeature(d, vnorm, unorm, self.h, self.w, c352, c1216, f352, f1216)
#         geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, self.h / 2, self.w / 2, c352, c1216, f352, f1216)
#         geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, self.h / 4, self.w / 4, c352, c1216, f352, f1216)
#         geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, self.h / 8, self.w / 8, c352, c1216, f352, f1216)
#         geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, self.h / 16, self.w / 16, c352, c1216, f352, f1216)
#         geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, self.h / 32, self.w / 32, c352, c1216, f352, f1216)

#         rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))
#         rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2)
#         rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2)
#         rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3)
#         rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3)
#         rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4)
#         rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4)
#         rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5)
#         rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5)
#         rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6)
#         rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6)

#         rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
#         rgb_feature8_plus = rgb_feature_decoder8 + rgb_feature8

#         rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
#         rgb_feature6_plus = rgb_feature_decoder6 + rgb_feature6

#         rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
#         rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4

#         rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
#         rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2

#         rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
#         rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature

#         rgb_output = self.rgb_decoder_output(rgb_feature0_plus)
#         rgb_depth = rgb_output[:, 0:1, :, :]

#         return rgb_depth