"""Explicit learning-rate groups, with stable ordering for existing checkpoints."""


def parameter_groups(model, settings):
    backbone, fresh = [], []
    prefixes = ("talker.model.layers.", "talker.model.norm.", "talker.model.embedding_norm.")
    if settings.get("text_embedding_lr_group", "fresh") == "backbone":
        prefixes += ("talker.model.text_embedding.", "talker.model.embed_tokens.")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            destination = backbone if name.startswith(prefixes) else fresh
            destination.append(parameter)
    return [
        {"params": backbone, "lr": settings["backbone_lr"]},
        {"params": fresh, "lr": settings["lr"]},
    ]
