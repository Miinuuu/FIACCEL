import math
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from model.flow_estimation import flow_estimation, flow_estimation_int
from model.selective_transfer import SelectiveTransfer
from model_pipeline import FIACCELPipeline

torch.set_num_threads(2)


class SelectiveTransferTests(unittest.TestCase):
    def test_whole_vector_and_retained_low_bits(self):
        cache = SelectiveTransfer()
        old = torch.full((2, 64, 2, 3), 64, dtype=torch.uint8)
        self.assertIsNone(cache.step(old))
        current = old ^ 31
        self.assertTrue(torch.equal(cache.step(current), old))
        self.assertTrue(torch.equal(cache.representative, old))
        current[1, 63, 1, 2] ^= 32
        self.assertTrue(torch.equal(cache.step(current), old))
        expected = old.clone()
        expected[1, 32:, 1, 2] = current[1, 32:, 1, 2]
        self.assertTrue(torch.equal(cache.representative, expected))
        self.assertEqual(cache.stats['written_vectors'], 25)
        self.assertEqual(cache.stats['skipped_vectors'], 47)
        self.assertEqual(cache.stats['written_bytes'], 25*32)

    def test_accumulated_drift_compares_to_retained_representative(self):
        cache = SelectiveTransfer()
        image = torch.zeros((1, 32, 1, 1), dtype=torch.uint8)
        cache.step(image)
        for low in range(1, 32):
            self.assertEqual(cache.step(image+low).sum().item(), 0)
        cache.step(image+32)
        self.assertEqual(cache.writes, 2)
        self.assertEqual(cache.skips, 31)
        self.assertTrue(torch.equal(cache.representative, image+32))
        cache.reset()
        self.assertIsNone(cache.representative)
        self.assertEqual(cache.reads, 0)

    def test_validation_and_disabled_baseline(self):
        cache = SelectiveTransfer(enabled=False)
        data = torch.zeros(1,32,1,1,dtype=torch.uint8)
        cache.step(data);cache.step(data)
        self.assertEqual((cache.writes,cache.skips), (2,0))
        with self.assertRaises(ValueError):cache.step(data.float())
        with self.assertRaises(ValueError):cache.step(torch.zeros(1,31,1,1,dtype=torch.uint8))
        with self.assertRaises(ValueError):cache.step(data.expand(2,-1,-1,-1))

    @unittest.skipUnless((ROOT/'sw/verification/verify.py').is_file(), 'Private plaintext RTL regression is not part of this package')
    def test_software_representatives_against_stalled_rtl(self):
        from sw.verification import verify
        frames = [torch.tensor([[64+c for c in range(32)], [128+c for c in range(32)]],
                               dtype=torch.uint8).T.reshape(1,32,1,2)]
        frames.append(frames[0]^31)
        frames.append(frames[1].clone());frames[2][0,31,0,0] ^= 32
        frames.append(frames[2]^7)
        cache = SelectiveTransfer();gold=[]
        for current in frames:
            old = cache.step(current)
            if old is not None:
                for x in range(2):
                    channels = old[0,:,0,x].tolist()+current[0,:,0,x].tolist()
                    gold.append(sum(v<<(8*c) for c,v in enumerate(channels)))
        expected_cache = [sum(v<<(8*c) for c,v in enumerate(cache.representative[0,:,0,x].tolist())) for x in range(2)]
        original_write = verify.write_hex
        def write(path, values, digits):
            if path.name == 'fmr_gold.hex':
                self.assertEqual(values,gold);values=gold
            if path.name == 'fmr_cache.hex':
                self.assertEqual(values,expected_cache);values=expected_cache
            original_write(path,values,digits)
        with tempfile.TemporaryDirectory() as directory, patch.object(verify,'write_hex',write):
            verify.selective(Path(directory))
        self.assertEqual((cache.writes,cache.skips), (3,5))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.images = torch.rand(1,3,16,24)

    def test_float_three_channel_hardtanh_and_training_gradient(self):
        model = flow_estimation()
        out = model(self.images,self.images*0.7)
        self.assertEqual(tuple(out.shape),(1,3,16,24))
        self.assertIsInstance(model.output[1],nn.Hardtanh)
        self.assertTrue(torch.all(out.abs()<=1))
        pipeline = FIACCELPipeline(model)
        prediction = pipeline.interpolate(self.images,self.images*0.7)
        prediction.mean().backward()
        self.assertTrue(torch.isfinite(model.enc[0].weight.grad).all())
        self.assertIsNone(model.feature_reuse.representative)

    def test_weight_export_defaults_to_three_channels(self):
        from sw.verification.export_weights import NAMES, shapes, normalize
        dimensions=shapes([8,16,32,64,32,16,8])
        self.assertEqual(dimensions[-1],(3,8,3,3))
        layers={name:dict(shape=shape,values=[0]*math.prod(shape),shift=8)
                for name,shape in zip(NAMES,dimensions)}
        _,normalized=normalize(dict(layers=layers))
        self.assertEqual(sum(len(weights) for _,weights,_ in normalized),194224)

    def test_stream_encodes_only_new_frame_and_resets(self):
        model = flow_estimation().eval()
        pipeline = FIACCELPipeline(model).eval()
        calls=[]
        hook=model.enc.register_forward_hook(lambda *args: calls.append(1))
        self.assertIsNone(pipeline.forward_frame(self.images))
        for _ in range(2):
            prediction=pipeline.forward_frame(self.images)
            self.assertEqual(prediction.shape,self.images.shape)
            self.assertTrue(torch.isfinite(prediction).all())
        hook.remove()
        self.assertEqual(len(calls),3)
        self.assertEqual((model.feature_reuse.writes,model.feature_reuse.skips),(6,12))
        self.assertFalse(any('representative' in key for key in model.state_dict()))
        pipeline.reset_stream()
        self.assertIsNone(pipeline.forward_frame(self.images))
        pipeline.train()
        self.assertIsNone(model.feature_reuse.representative)

    def test_independent_clip_inference_and_cpu(self):
        pipeline=FIACCELPipeline(flow_estimation()).eval()
        clip=SimpleNamespace(video_tensor=torch.stack([self.images,self.images,self.images],dim=1))
        first=pipeline.inference(clip)
        second=pipeline.inference(clip)
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(first.device.type,'cpu')
        self.assertEqual(pipeline.flow_estimation.feature_reuse.reads,6)

    def test_midpoint_and_scale_guards(self):
        model=flow_estimation().eval()
        with self.assertRaises(ValueError):model.forward_frame(self.images,timestep=0.25)
        model.forward_frame(self.images)
        model.feature_scale *= 2
        with self.assertRaises(ValueError):model.forward_frame(self.images)
        model.reset_stream()
        self.assertIsNone(model.forward_frame(self.images))
        model.train()
        with self.assertRaises(RuntimeError):model.forward_frame(self.images)

    @unittest.skipUnless(importlib.util.find_spec('brevitas'), 'Brevitas is required for QAT tests')
    def test_quantized_three_channels_and_actual_encoder_codes(self):
        model=flow_estimation_int()
        out=model(self.images,self.images*0.8)
        self.assertEqual(tuple(out.shape),(1,3,16,24))
        self.assertIsInstance(model.output[1],nn.Hardtanh)
        model.eval()
        encoded=model._encode(self.images)
        expected=encoded.int(float_datatype=False).to(torch.uint8)
        self.assertIsNone(model.forward_frame(self.images))
        self.assertTrue(torch.equal(model.feature_reuse.representative,expected))
        out=model.forward_frame(self.images)
        self.assertEqual(tuple(out.shape),(1,3,16,24))
        self.assertEqual(model.feature_reuse.skips,6)

    def test_synthesis_geometry_mask_and_border_against_scalar_bilinear(self):
        previous=torch.rand(1,3,3,5);current=torch.rand_like(previous)
        out=torch.zeros(1,3,3,5);out[:,0]=0.37;out[:,1]=-0.23;out[:,2]=0.41
        actual=FIACCELPipeline.synthesize(previous,current,out)
        def sample(image,c,x,y):
            x=max(0,min(4,x));y=max(0,min(2,y));x0=int(math.floor(x));y0=int(math.floor(y))
            x1=min(4,x0+1);y1=min(2,y0+1);fx=x-x0;fy=y-y0
            return sum(float(image[0,c,yy,xx])*weight for yy,xx,weight in
                       ((y0,x0,(1-fx)*(1-fy)),(y0,x1,fx*(1-fy)),
                        (y1,x0,(1-fx)*fy),(y1,x1,fx*fy)))
        for c in range(3):
            for y in range(3):
                for x in range(5):
                    a=sample(previous,c,x-0.37,y+0.23*2/4)
                    b=sample(current,c,x+0.37,y-0.23*2/4)
                    self.assertAlmostEqual(actual[0,c,y,x].item(),a*0.705+b*0.295,places=6)
        zero=torch.zeros_like(out)
        self.assertTrue(torch.allclose(FIACCELPipeline.synthesize(previous,current,zero),(previous+current)/2,atol=1e-6))
        with self.assertRaises(ValueError):FIACCELPipeline.synthesize(previous,current,torch.zeros(1,5,3,5))


if __name__=='__main__':unittest.main()
