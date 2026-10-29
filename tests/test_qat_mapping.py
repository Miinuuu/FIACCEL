"""Cross-check deployment arithmetic against the independent scalar oracle."""
import random
import unittest
import torch

from sw.verification.map_qat_weights import fixed_coefficients, IntegerNetwork, synthesize
from sw.verification.scalar_reference import reference
from sw.verification.scalar_reference import synthesize as scalar_synthesize


class MappingTests(unittest.TestCase):
    def test_shift_precision_and_signed_limits(self):
        x = torch.tensor([0.1, -0.12, 0.2], dtype=torch.float64)
        q, shift, error = fixed_coefficients(x)
        self.assertEqual(shift, 9)
        self.assertTrue(((q.double()/(2**shift)-x).abs() <= 0.5/(2**shift)).all())
        self.assertGreater(error, 0)
        self.assertEqual(fixed_coefficients(torch.tensor([-128.]))[0].item(), -128)
        self.assertEqual(fixed_coefficients(torch.zeros(3))[1], 0)
        for invalid in (torch.tensor([128.]), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                fixed_coefficients(invalid)

    def test_convolution_layout_dilation_transpose_floor_and_head(self):
        rng = random.Random(731)
        for name, ci, co, k, stride, dilation, transpose, head in (
                ("enc.0", 3, 2, 3, 2, 1, False, False),
                ("aspp2.0", 2, 3, 3, 1, 2, False, False),
                ("shrink.0", 2, 3, 1, 1, 1, False, False),
                ("dec.0", 2, 3, 3, 1, 1, True, False),
                ("output.0", 2, 3, 3, 1, 1, False, True)):
            with self.subTest(layer=name):
                image = [[[rng.randrange(-128 if name == "enc.0" else 0, 128 if name == "enc.0" else 256)
                           for _ in range(ci)] for _ in range(5)] for _ in range(3)]
                weight = torch.tensor([rng.randrange(-128, 128) for _ in range(co*ci*k*k)])
                weight = weight.reshape(co, ci, k, k)
                source = weight.permute(1, 0, 2, 3).contiguous() if transpose else weight
                shift = 0 if head else 4
                document = {"layers": {name: dict(values=source.flatten().tolist(), shape=list(source.shape), shift=shift)}}
                net = IntegerNetwork(document)
                actual = net.layer(torch.tensor(image).permute(2, 0, 1).unsqueeze(0).double(), name)
                gold = reference(image, weight.flatten().tolist(), co, k, stride, dilation, transpose, shift, head)
                expected = torch.tensor(gold).clamp(-16384, 16384) if head else torch.tensor(gold)
                self.assertTrue(torch.equal(actual[0].permute(1, 2, 0).long(), expected))

    def test_q8_warp_blend_and_borders_against_scalar_oracle(self):
        rng = random.Random(27)
        frames = [[[[rng.randrange(256) for _ in range(3)] for _ in range(5)] for _ in range(3)] for _ in range(2)]
        flow = [[[rng.randrange(-16384, 16385) for _ in range(3)] for _ in range(5)] for _ in range(3)]
        tensors = [torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) for frame in frames]
        actual = synthesize(*tensors, torch.tensor(flow).permute(2, 0, 1).unsqueeze(0))
        expected = torch.tensor(scalar_synthesize(*frames, flow))
        self.assertTrue(torch.equal(actual[0].permute(1, 2, 0), expected))


if __name__ == "__main__":
    unittest.main()
