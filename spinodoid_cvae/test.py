import unittest
import torch
from models.decoder import Decoder
from config import S_DIM, P_DIM, LATENT_DIM, DECODER_HIDDEN_DIMS

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.S_dim = S_DIM
        self.P_dim = P_DIM
        self.latent_dim = LATENT_DIM
        self.batch_size = 8
        self.decoder = Decoder(self.S_dim, self.P_dim, self.latent_dim)

    def test_init(self):
        self.assertIsInstance(self.decoder, Decoder)
        self.assertTrue(hasattr(self.decoder, 'hidden_layers'))
        self.assertTrue(hasattr(self.decoder, 'output_layer'))

    def test_forward_output_shape(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        P = torch.randn(self.batch_size, self.P_dim)
        S_hat = self.decoder(z, P)
        self.assertEqual(S_hat.shape, (self.batch_size, self.S_dim))

    def test_forward_batch_size_1(self):
        z = torch.randn(1, self.latent_dim)
        P = torch.randn(1, self.P_dim)
        S_hat = self.decoder(z, P)
        self.assertEqual(S_hat.shape, (1, self.S_dim))

    def test_forward_mismatched_dimensions(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        P = torch.randn(self.batch_size, self.P_dim + 1)  # wrong dim
        with self.assertRaises(RuntimeError):
            self.decoder(z, P)

    def test_device_compatibility(self):
        if torch.cuda.is_available():
            decoder_cuda = Decoder(self.S_dim, self.P_dim, self.latent_dim).cuda()
            z = torch.randn(self.batch_size, self.latent_dim, device='cuda')
            P = torch.randn(self.batch_size, self.P_dim, device='cuda')
            S_hat = decoder_cuda(z, P)
            self.assertEqual(S_hat.device.type, 'cuda')
        else:
            self.skipTest('CUDA not available')

    def test_parameters_require_grad(self):
        for param in self.decoder.parameters():
            self.assertTrue(param.requires_grad)

    def test_hidden_layer_config(self):
        # Check number of layers (Linear+ReLU pairs)
        num_layers = sum(1 for m in self.decoder.hidden_layers if isinstance(m, torch.nn.Linear))
        self.assertEqual(num_layers, len(DECODER_HIDDEN_DIMS))

    def test_output_type(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        P = torch.randn(self.batch_size, self.P_dim)
        S_hat = self.decoder(z, P)
        self.assertIsInstance(S_hat, torch.Tensor)

    def test_reproducibility(self):
        torch.manual_seed(42)
        decoder1 = Decoder(self.S_dim, self.P_dim, self.latent_dim)
        z = torch.randn(self.batch_size, self.latent_dim)
        P = torch.randn(self.batch_size, self.P_dim)
        out1 = decoder1(z, P)
        torch.manual_seed(42)
        decoder2 = Decoder(self.S_dim, self.P_dim, self.latent_dim)
        z2 = torch.randn(self.batch_size, self.latent_dim)
        P2 = torch.randn(self.batch_size, self.P_dim)
        out2 = decoder2(z2, P2)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
