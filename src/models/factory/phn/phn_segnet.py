# Adapted from https://github.com/ruchtem/cosmos and https://github.com/AvivNavon/pareto-hypernetworks
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.factory.segnet_cityscapes import SegNet


def measure_target_network_params(params):
    # for sanity check
    num_params = sum([v.numel() for k, v in params.items()])
    print("NUM_PARAMS=", num_params)


class SegNetHyper(nn.Module):
    def __init__(
        self,
        preference_dim=2,
        preference_embedding_dim=32,
        hidden_dim=100,
        num_chunks=105,
        chunk_embedding_dim=64,
        num_ws=24,
        w_dim=20000,
    ):
        """
        :param preference_dim: preference vector dimension
        :param preference_embedding_dim: preference embedding dimension
        :param hidden_dim: hidden dimension
        :param num_chunks: number of chunks
        :param chunk_embedding_dim: chunks embedding dimension
        :param num_ws: number of W matrices (see paper for details)
        :param w_dim: row dimension of the W matrices
        """
        super().__init__()
        self.preference_embedding_dim = preference_embedding_dim
        self.num_chunks = num_chunks
        self.chunk_embedding_matrix = nn.Embedding(num_embeddings=num_chunks, embedding_dim=chunk_embedding_dim)
        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        list_ws = [self._init_w((w_dim, hidden_dim)) for _ in range(num_ws)]
        self.ws = nn.ParameterList(list_ws)

        # initialization
        torch.nn.init.normal_(self.preference_embedding_matrix.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)
        for w in self.ws:
            torch.nn.init.normal_(w, mean=0.0, std=0.1)

        self.layer_to_shape = {
            "segnet.encoder_block.0.0.weight": torch.Size([64, 3, 3, 3]),
            "segnet.encoder_block.0.0.bias": torch.Size([64]),
            "segnet.encoder_block.0.1.weight": torch.Size([64]),
            "segnet.encoder_block.0.1.bias": torch.Size([64]),
            "segnet.encoder_block.1.0.weight": torch.Size([128, 64, 3, 3]),
            "segnet.encoder_block.1.0.bias": torch.Size([128]),
            "segnet.encoder_block.1.1.weight": torch.Size([128]),
            "segnet.encoder_block.1.1.bias": torch.Size([128]),
            "segnet.encoder_block.2.0.weight": torch.Size([256, 128, 3, 3]),
            "segnet.encoder_block.2.0.bias": torch.Size([256]),
            "segnet.encoder_block.2.1.weight": torch.Size([256]),
            "segnet.encoder_block.2.1.bias": torch.Size([256]),
            "segnet.encoder_block.3.0.weight": torch.Size([512, 256, 3, 3]),
            "segnet.encoder_block.3.0.bias": torch.Size([512]),
            "segnet.encoder_block.3.1.weight": torch.Size([512]),
            "segnet.encoder_block.3.1.bias": torch.Size([512]),
            "segnet.encoder_block.4.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.encoder_block.4.0.bias": torch.Size([512]),
            "segnet.encoder_block.4.1.weight": torch.Size([512]),
            "segnet.encoder_block.4.1.bias": torch.Size([512]),
            "segnet.decoder_block.0.0.weight": torch.Size([64, 64, 3, 3]),
            "segnet.decoder_block.0.0.bias": torch.Size([64]),
            "segnet.decoder_block.0.1.weight": torch.Size([64]),
            "segnet.decoder_block.0.1.bias": torch.Size([64]),
            "segnet.decoder_block.1.0.weight": torch.Size([64, 128, 3, 3]),
            "segnet.decoder_block.1.0.bias": torch.Size([64]),
            "segnet.decoder_block.1.1.weight": torch.Size([64]),
            "segnet.decoder_block.1.1.bias": torch.Size([64]),
            "segnet.decoder_block.2.0.weight": torch.Size([128, 256, 3, 3]),
            "segnet.decoder_block.2.0.bias": torch.Size([128]),
            "segnet.decoder_block.2.1.weight": torch.Size([128]),
            "segnet.decoder_block.2.1.bias": torch.Size([128]),
            "segnet.decoder_block.3.0.weight": torch.Size([256, 512, 3, 3]),
            "segnet.decoder_block.3.0.bias": torch.Size([256]),
            "segnet.decoder_block.3.1.weight": torch.Size([256]),
            "segnet.decoder_block.3.1.bias": torch.Size([256]),
            "segnet.decoder_block.4.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.decoder_block.4.0.bias": torch.Size([512]),
            "segnet.decoder_block.4.1.weight": torch.Size([512]),
            "segnet.decoder_block.4.1.bias": torch.Size([512]),
            "segnet.conv_block_enc.0.0.weight": torch.Size([64, 64, 3, 3]),
            "segnet.conv_block_enc.0.0.bias": torch.Size([64]),
            "segnet.conv_block_enc.0.1.weight": torch.Size([64]),
            "segnet.conv_block_enc.0.1.bias": torch.Size([64]),
            "segnet.conv_block_enc.1.0.weight": torch.Size([128, 128, 3, 3]),
            "segnet.conv_block_enc.1.0.bias": torch.Size([128]),
            "segnet.conv_block_enc.1.1.weight": torch.Size([128]),
            "segnet.conv_block_enc.1.1.bias": torch.Size([128]),
            "segnet.conv_block_enc.2.0.0.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_enc.2.0.0.bias": torch.Size([256]),
            "segnet.conv_block_enc.2.0.1.weight": torch.Size([256]),
            "segnet.conv_block_enc.2.0.1.bias": torch.Size([256]),
            "segnet.conv_block_enc.2.1.0.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_enc.2.1.0.bias": torch.Size([256]),
            "segnet.conv_block_enc.2.1.1.weight": torch.Size([256]),
            "segnet.conv_block_enc.2.1.1.bias": torch.Size([256]),
            "segnet.conv_block_enc.3.0.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.3.0.0.bias": torch.Size([512]),
            "segnet.conv_block_enc.3.0.1.weight": torch.Size([512]),
            "segnet.conv_block_enc.3.0.1.bias": torch.Size([512]),
            "segnet.conv_block_enc.3.1.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.3.1.0.bias": torch.Size([512]),
            "segnet.conv_block_enc.3.1.1.weight": torch.Size([512]),
            "segnet.conv_block_enc.3.1.1.bias": torch.Size([512]),
            "segnet.conv_block_enc.4.0.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.4.0.0.bias": torch.Size([512]),
            "segnet.conv_block_enc.4.0.1.weight": torch.Size([512]),
            "segnet.conv_block_enc.4.0.1.bias": torch.Size([512]),
            "segnet.conv_block_enc.4.1.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.4.1.0.bias": torch.Size([512]),
            "segnet.conv_block_enc.4.1.1.weight": torch.Size([512]),
            "segnet.conv_block_enc.4.1.1.bias": torch.Size([512]),
            "segnet.conv_block_dec.0.0.weight": torch.Size([64, 64, 3, 3]),
            "segnet.conv_block_dec.0.0.bias": torch.Size([64]),
            "segnet.conv_block_dec.0.1.weight": torch.Size([64]),
            "segnet.conv_block_dec.0.1.bias": torch.Size([64]),
            "segnet.conv_block_dec.1.0.weight": torch.Size([64, 64, 3, 3]),
            "segnet.conv_block_dec.1.0.bias": torch.Size([64]),
            "segnet.conv_block_dec.1.1.weight": torch.Size([64]),
            "segnet.conv_block_dec.1.1.bias": torch.Size([64]),
            "segnet.conv_block_dec.2.0.0.weight": torch.Size([128, 128, 3, 3]),
            "segnet.conv_block_dec.2.0.0.bias": torch.Size([128]),
            "segnet.conv_block_dec.2.0.1.weight": torch.Size([128]),
            "segnet.conv_block_dec.2.0.1.bias": torch.Size([128]),
            "segnet.conv_block_dec.2.1.0.weight": torch.Size([128, 128, 3, 3]),
            "segnet.conv_block_dec.2.1.0.bias": torch.Size([128]),
            "segnet.conv_block_dec.2.1.1.weight": torch.Size([128]),
            "segnet.conv_block_dec.2.1.1.bias": torch.Size([128]),
            "segnet.conv_block_dec.3.0.0.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_dec.3.0.0.bias": torch.Size([256]),
            "segnet.conv_block_dec.3.0.1.weight": torch.Size([256]),
            "segnet.conv_block_dec.3.0.1.bias": torch.Size([256]),
            "segnet.conv_block_dec.3.1.0.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_dec.3.1.0.bias": torch.Size([256]),
            "segnet.conv_block_dec.3.1.1.weight": torch.Size([256]),
            "segnet.conv_block_dec.3.1.1.bias": torch.Size([256]),
            "segnet.conv_block_dec.4.0.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_dec.4.0.0.bias": torch.Size([512]),
            "segnet.conv_block_dec.4.0.1.weight": torch.Size([512]),
            "segnet.conv_block_dec.4.0.1.bias": torch.Size([512]),
            "segnet.conv_block_dec.4.1.0.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_dec.4.1.0.bias": torch.Size([512]),
            "segnet.conv_block_dec.4.1.1.weight": torch.Size([512]),
            "segnet.conv_block_dec.4.1.1.bias": torch.Size([512]),
            "segnet.pred_task1.0.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task1.0.bias": torch.Size([64]),
            "segnet.pred_task1.1.weight": torch.Size([7, 64, 1, 1]),
            "segnet.pred_task1.1.bias": torch.Size([7]),
            "segnet.pred_task2.0.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task2.0.bias": torch.Size([64]),
            "segnet.pred_task2.1.weight": torch.Size([1, 64, 1, 1]),
            "segnet.pred_task2.1.bias": torch.Size([1]),
        }

    def _init_w(self, shapes):
        return nn.Parameter(torch.randn(shapes), requires_grad=True)

    def forward(self, preference):
        # preference embedding
        device = preference.device
        pref_embedding = torch.zeros((self.preference_embedding_dim,), device=device)
        for i, pref in enumerate(preference):
            pref_embedding += self.preference_embedding_matrix(torch.tensor([i], device=device)).squeeze(0) * pref
        # chunk embedding
        weights = []
        for chunk_id in range(self.num_chunks):
            chunk_embedding = self.chunk_embedding_matrix(torch.tensor([chunk_id], device=device)).squeeze(0)
            # input to fc
            input_embedding = torch.cat((pref_embedding, chunk_embedding)).unsqueeze(0)
            # hidden representation
            rep = self.fc(input_embedding)

            weights.append(torch.cat([F.linear(rep, weight=w) for w in self.ws], dim=1))

        weight_vector = torch.cat(weights, dim=1).squeeze(0)
        out_dict = dict()
        position = 0
        for name, shapes in self.layer_to_shape.items():
            out_dict[name] = weight_vector[position : position + shapes.numel()].reshape(shapes)
            position += shapes.numel()

        return out_dict


class SegNetTarget(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        self.segnet = SegNet()

    def forward(self, x, weights=None):
        if weights is not None:
            for name, param in self.segnet.named_parameters():
                param.data = weights[name]

        return self.segnet(x)
