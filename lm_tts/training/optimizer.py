"""Model-owned learning-rate groups with checkpoint-stable parameter ordering."""


def parameter_groups(model, settings):
    return model.parameter_groups(settings)
