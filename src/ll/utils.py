from src.ll.subspace_modules import SubspaceConv, SubspaceModule, SubspaceLinear
import torch


def extract_model(module0, module1):
    if isinstance(module0, SubspaceModule):
        assert isinstance(module0, (SubspaceLinear, SubspaceConv)), "add code for other types of subspace modules"
        w, b = module0.get_weight()
        with torch.no_grad():
            module1.weight.copy_(w)
            module1.bias.copy_(b)
    else:
        for m0, m1 in zip(module0.children(), module1.children()):
            extract_model(m0, m1)


def set_member(module0, module1, member_id):
    if isinstance(module0, SubspaceModule):
        assert isinstance(module0, (SubspaceLinear, SubspaceConv)), "add code for other types of subspace modules"
        w, b = module1.weight, module1.bias
        with torch.no_grad():
            module0.members[member_id].weight.copy_(w)
            module0.members[member_id].bias.copy_(b)
    else:
        for m0, m1 in zip(module0.children(), module1.children()):
            set_member(m0, m1, member_id)


def freeze_member(module, member_id):
    if isinstance(module, SubspaceModule):
        member_module = module.members[member_id]
        for param in member_module.parameters():
            param.requires_grad = False
    else:
        for m in module.children():
            freeze_member(m, member_id)
