import torch.nn as nn
from src.models.base_model import SharedBottom


class SimpleMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        nonlinearity="relu",
        remove_last_activation=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.remove_last_activation = remove_last_activation

        feats = [in_features] + list(hidden_features)
        self.feats = feats
        if nonlinearity == "relu":
            nonlinearity_fn = nn.ReLU()
        else:
            raise NotImplementedError

        modules = []
        for inf, outf in zip(feats[:-1], feats[1:]):
            modules.append(nn.Linear(in_features=inf, out_features=outf))
            modules.append(nonlinearity_fn)

        if remove_last_activation:
            # remove last activation
            modules = modules[:-1]

        self.seq = nn.Sequential(*modules)

    def get_last_layer(self):
        index = -1 - int(not self.remove_last_activation)
        return self.seq[index]

    def forward(self, x):
        return self.seq(x)

    def reset_parameters(self):
        for m in self.seq.children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


class MultiTaskMLP(SharedBottom):
    def __init__(self, in_features, num_tasks, encoder_specs, decoder_specs):
        self.in_features = in_features
        self.encoder_specs = encoder_specs
        self.decoder_specs = decoder_specs
        encoder = SimpleMLP(in_features=in_features, hidden_features=encoder_specs, remove_last_activation=False)
        decoder = SimpleMLP(in_features=encoder_specs[-1], hidden_features=decoder_specs)
        super().__init__(encoder, decoder, num_tasks)


class MultiTaskMLPDifferentNumClasses(SharedBottom):
    def __init__(self, in_features, num_tasks, encoder_specs, decoder_specs, num_classes):
        assert num_tasks == len(num_classes)
        self.in_features = in_features
        self.encoder_specs = encoder_specs
        self.decoder_specs = decoder_specs
        encoder = SimpleMLP(in_features=in_features, hidden_features=encoder_specs, remove_last_activation=False)

        decoders = []
        for c in num_classes:
            decoder = SimpleMLP(in_features=encoder_specs[-1], hidden_features=decoder_specs + [c])
            decoders.append(decoder)
        super().__init__(encoder, decoders, num_tasks)
