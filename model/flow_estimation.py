import torch
import torch.nn as nn
from .selective_transfer import SelectiveTransfer
    
class _FeatureStream(nn.Module):
    """Inference-only temporal encoder reuse with the same FMR rule as RTL."""

    def _init_stream(self, feature_scale, selective_transfer):
        if not 0 < feature_scale < float('inf'):
            raise ValueError('feature_scale must be finite and positive')
        self.register_buffer('feature_scale', torch.tensor(float(feature_scale)), persistent=False)
        self.register_buffer('_stream_scale', None, persistent=False)
        self.feature_reuse = SelectiveTransfer(enabled=selective_transfer)
        self._stream_input_shape = None

    def reset_stream(self):
        self.feature_reuse.reset()
        self._stream_scale = None
        self._stream_input_shape = None

    @staticmethod
    def _value(tensor):
        return tensor if isinstance(tensor, torch.Tensor) else tensor.value

    def _encode(self, image):
        if hasattr(self, 'quant_input'):
            image = self.quant_input(image)
        return self.enc(image)

    def _decode(self, previous, current):
        combined = torch.cat([previous, current], dim=1)
        return self._value(self.output(self.dec(self.bottleneck(combined))))

    def _feature_codes(self, encoded):
        value = self._value(encoded)
        if isinstance(encoded, torch.Tensor):
            scale = self.feature_scale.to(value)
            codes = torch.floor(value / scale).clamp(0, 255).to(torch.uint8)
        else:
            if encoded.scale.numel() != 1 or encoded.signed or encoded.bit_width.item() != 8:
                raise ValueError('FMR requires per-tensor unsigned 8-bit encoder quantization')
            if torch.any(encoded.zero_point != 0):
                raise ValueError('FMR requires a zero encoder quantization zero-point')
            scale = encoded.scale.detach().to(value)
            codes = encoded.int(float_datatype=False).clamp(0, 255).to(torch.uint8)
        if self._stream_scale is not None and not torch.equal(scale, self._stream_scale):
            raise ValueError('Encoder feature scale changed within a stream; freeze quantization and reset_stream()')
        return codes, scale

    @torch.no_grad()
    def forward_frame(self, image, timestep=0.5):
        """Encode one new RGB frame; warm-up returns None, then three channels.

        Float features use the explicit calibrated ``feature_scale``. Brevitas
        uses its encoder's actual UINT8 codes and fixed evaluation scale. This
        models FMR transfers; floating/QAT convolutions are not an integer MAC
        simulator. For RTL arithmetic use the existing integer RTL reference.
        """
        if self.training:
            raise RuntimeError('Streaming inference requires model.eval()')
        if timestep != 0.5:
            raise ValueError('FIACCEL supports midpoint interpolation (timestep=0.5)')
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected RGB frames with shape [B,3,H,W]')
        if self._stream_input_shape is not None and image.shape != self._stream_input_shape:
            raise ValueError('Frame shape changed; call reset_stream() first')
        encoded = self._encode(image)
        codes, scale = self._feature_codes(encoded)
        old = self.feature_reuse.step(codes)
        self._stream_scale = scale.clone()
        self._stream_input_shape = image.shape
        if old is None:
            return None
        dtype = self._value(encoded).dtype
        return self._decode(old.to(dtype) * scale, codes.to(dtype) * scale)

    def train(self, mode=True):
        if mode and hasattr(self, 'feature_reuse'):
            self.reset_stream()
        return super().train(mode)


class flow_estimation(_FeatureStream):
    def __init__(self, 
                 depths = [8, 16, 32 , 64, 32 ,16 ,8],
                 feature_scale=1.0/255.0, selective_transfer=True,
                 ):
        super().__init__()
        self._init_stream(feature_scale, selective_transfer)
       

        self.enc =torch.nn.Sequential(  nn.Conv2d(3, depths[0], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU(),

                                        nn.Conv2d(depths[0], depths[1], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU(),

                                        nn.Conv2d(depths[1], depths[2],  stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU())

        self.aspp1 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=1, kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU(),)

        self.aspp2 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,2), kernel_size=(3,3), padding=(1,2), bias=False),
                                        nn.ReLU(),)
                                        
        self.aspp3 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,3), kernel_size=(3,3), padding=(1,3), bias=False),
                                        nn.ReLU(),)
        self.aspp4 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,4), kernel_size=(3,3), padding=(1,4), bias=False),
                                        nn.ReLU(),)

        self.shrink = torch.nn.Sequential(  nn.Conv2d(depths[3]*4, depths[3], stride=(1, 1), dilation=1, kernel_size=(1,1), padding=(0,0), bias=False),
                                            nn.ReLU(),
        )

        self.dec =torch.nn.Sequential (
                                nn.ConvTranspose2d(depths[3], depths[4], 3, stride=2, padding=1, bias=False, output_padding=1),nn.ReLU(),
                                nn.ConvTranspose2d(depths[4], depths[5], 3, stride=2, padding=1, bias=False, output_padding=1),nn.ReLU(),
                                nn.ConvTranspose2d(depths[5], depths[6],  3, stride=2, padding=1, bias=False, output_padding=1),nn.ReLU()
        )

        self.output = torch.nn.Sequential(  nn.Conv2d(depths[6], 3, stride=(1, 1), kernel_size=(3,3), padding=1, bias=False),
                                            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )         
    def bottleneck(self,FM) :
        fm1= self.aspp1(FM)
        fm2= self.aspp2(FM)
        fm3= self.aspp3(FM)
        fm4= self.aspp4(FM)
        return self.shrink(torch.cat([fm1,fm2,fm3,fm4],dim=1))

    def forward(self,img0,img1, timestep=0.5):
        
        B,C,H,W= img0.shape
        img_cat = torch.cat([img0,img1],dim=0)

        encoder_FM = self.enc(img_cat)
        encoder_FM0=encoder_FM[:B,...]
        encoder_FM1=encoder_FM[B:,...]
        encoder_FM =  torch.cat([encoder_FM0,encoder_FM1],dim=1)
        bottleneck_FM = self.bottleneck(encoder_FM)
        dec_FM = self.dec(bottleneck_FM)
        out =    self.output(dec_FM)
        
        del img_cat
        del encoder_FM
        del encoder_FM0
        del encoder_FM1
        del bottleneck_FM
        del dec_FM

        return out


try:
    import brevitas.nn as qnn
    from brevitas.nn import QuantIdentity
except ImportError:
    qnn = None
    QuantIdentity = None

class flow_estimation_int(_FeatureStream):
    def __init__(self, 
                 depths = [8, 16, 32 , 64, 32 ,16 ,8],
                 feature_scale=1.0/255.0, selective_transfer=True,
                 ):
        super().__init__()
        self._init_stream(feature_scale, selective_transfer)
       
        if qnn is None:
            raise ImportError("flow_estimation_int requires brevitas")
        self.quant_input = QuantIdentity(bit_width=8, return_quant_tensor=True)

        self.enc =torch.nn.Sequential(  qnn.QuantConv2d(3, depths[0], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(),

                                        qnn.QuantConv2d(depths[0], depths[1], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(),

                                        qnn.QuantConv2d(depths[1], depths[2],  stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(bit_width=8, return_quant_tensor=True))

        self.aspp1 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=1, kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(),)

        self.aspp2 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,2), kernel_size=(3,3), padding=(1,2), bias=False),
                                        qnn.QuantReLU(),)
                                        
        self.aspp3 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,3), kernel_size=(3,3), padding=(1,3), bias=False),
                                        qnn.QuantReLU(),)
        self.aspp4 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,4), kernel_size=(3,3), padding=(1,4), bias=False),
                                        qnn.QuantReLU(),)

        self.shrink = torch.nn.Sequential(  qnn.QuantConv2d(depths[3]*4, depths[3], stride=(1, 1), dilation=1, kernel_size=(1,1), padding=(0,0), bias=False),
                                            qnn.QuantReLU(),
        )

        self.dec =torch.nn.Sequential (
                                qnn.QuantConvTranspose2d(depths[3], depths[4], 3, stride=2, padding=1, bias=False, output_padding=1),qnn.QuantReLU(),
                                qnn.QuantConvTranspose2d(depths[4], depths[5], 3, stride=2, padding=1, bias=False, output_padding=1),qnn.QuantReLU(),
                                qnn.QuantConvTranspose2d(depths[5], depths[6],  3, stride=2, padding=1, bias=False, output_padding=1),qnn.QuantReLU()
        )

        self.output = torch.nn.Sequential(  qnn.QuantConv2d(depths[6], 3, stride=(1, 1), kernel_size=(3,3), padding=1, bias=False),
                                            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )         
    def bottleneck(self,FM) :
        fm1= self.aspp1(FM)
        fm2= self.aspp2(FM)
        fm3= self.aspp3(FM)
        fm4= self.aspp4(FM)
        return self.shrink(torch.cat([self._value(fm1),self._value(fm2),self._value(fm3),self._value(fm4)],dim=1))

    def forward(self,img0,img1, timestep=0.5):
        
        B,C,H,W= img0.shape
        encoder_FM0 = self.enc(self.quant_input(img0))
        encoder_FM1 = self.enc(self.quant_input(img1))

        encoder_FM =  torch.cat([self._value(encoder_FM0),self._value(encoder_FM1)],dim=1)
        bottleneck_FM = self.bottleneck(encoder_FM)
        dec_FM = self.dec(bottleneck_FM)
        out =    self.output(dec_FM)
        
        del encoder_FM
        del encoder_FM0
        del encoder_FM1
        del bottleneck_FM
        del dec_FM

        return out

