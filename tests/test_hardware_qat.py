import io
import unittest

import torch
from torch import nn

from model.hardware_qat import HardwareConv, HardwareFlow, HardwarePipeline, integer_ste


class HardwareQATTests(unittest.TestCase):
    def test_floor_and_round_keep_nonzero_ste_gradients(self):
        x = torch.tensor([-1.7, -0.1, 0.2, 1.2], requires_grad=True)
        y = integer_ste(x, "floor")
        torch.testing.assert_close(y, torch.tensor([-2., -1., 0., 1.]), rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_integer_weights_and_shift_survive_reload(self):
        layer = HardwareConv(1, 2, k=1, shift=4, head=True).eval()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([12., -20.]).reshape(2, 1, 1, 1)/16)
        self.assertIsNone(layer.bias)
        self.assertEqual(layer.integer_weight().flatten().tolist(), [12., -20.])
        x = torch.tensor([0., 1., 10., 255.]).reshape(1, 1, 1, 4)
        expected = torch.tensor([0., 0., 7., 191., 0., -2., -13., -319.]).reshape(1, 2, 1, 4)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(layer.state_dict(), stream)
        stream.seek(0)
        restored = HardwareConv(1, 2, k=1, shift=1, head=True).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertEqual(restored.shift, 4)
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_weight_backward_without_extra_parameters(self):
        layer = HardwareConv(1, 2, k=1, shift=4, head=True).train()
        with torch.no_grad():
            layer.weight.fill_(0.5)
        output = layer(torch.ones(2, 1, 3, 3)*10)
        output.sum().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        model = HardwareFlow()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 194224)
        self.assertEqual(len(list(model.parameters())), 12)
        self.assertTrue(all(name.endswith(".weight") for name, _ in model.named_parameters()))

    def test_integer_rgb_forward_retains_flow_gradients(self):
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.flow = nn.Parameter(torch.zeros(1, 3, 8, 16))

            def forward(self, a, b):
                return self.flow

        torch.manual_seed(93)
        pipeline = HardwarePipeline(Head())
        a, b = torch.randint(0, 256, (2, 1, 3, 8, 16)).float()/255
        prediction = pipeline.interpolate(a, b)
        torch.testing.assert_close((prediction*255).round(), (((a*255).round()+(b*255).round()+1)/2).floor(), rtol=0, atol=0)
        prediction.square().mean().backward()
        gradient = pipeline.flow_estimation.flow.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[:, :2].abs().sum().item(), 0)
        self.assertGreater(gradient[:, 2:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
