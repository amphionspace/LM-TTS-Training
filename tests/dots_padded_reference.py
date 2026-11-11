"""Pre-packing DiT implementation, retained only as an equivalence oracle."""

import torch

CHUNK = 5


def padded_velocity(self, hidden, clean, noisy, t, condition, patch_lengths):
    """Predicted velocity for every noisy patch, packed (sum N, 4, D).

    Per utterance the DiT sees a causal prefix [h_k, clean_k] for k = 1..N and, for each
    patch, a generation chunk [h_k, noisy_k] that attends to itself and to the prefix chunks
    before k. Prefix and generation chunks of patch k share rotary positions. Utterances are
    right-padded; padded queries attend only to themselves."""
    p, n = self.patch_size, int(patch_lengths.max())
    valid = torch.arange(n, device=hidden.device) < patch_lengths[:, None]

    def chunks(latents, projection):
        frames = latents.new_zeros(len(patch_lengths), n, p, latents.shape[-1])
        frames[valid] = latents.view(-1, p, latents.shape[-1])
        values = hidden.new_zeros(len(patch_lengths), n, 1, hidden.shape[-1])
        values[valid] = hidden[:, None]
        return torch.cat([values, projection(frames.to(hidden.dtype))], dim=2).flatten(1, 2)

    sequence = torch.cat(
        [chunks(clean, self.latent_proj), chunks(noisy, self.coordinate_proj)], dim=1
    )
    index = torch.arange(2 * CHUNK * n, device=hidden.device)
    generation, local = index >= CHUNK * n, index % (CHUNK * n)
    chunk = local // CHUNK
    prefix_causal = ~generation[:, None] & ~generation[None] & (local[None] <= local[:, None])
    history = generation[:, None] & ~generation[None] & (local[None] < CHUNK * chunk[:, None])
    own = generation[:, None] & generation[None] & (chunk[None] == chunk[:, None])
    allowed = valid[:, chunk]
    mask = (prefix_causal | history | own)[None] & allowed[:, None, :] & allowed[:, :, None]
    mask = mask | (
        ~allowed[:, :, None] & torch.eye(len(index), dtype=torch.bool, device=index.device)
    )
    predictor = self.velocity_field_predictor
    c = predictor.time_embedder(t) + condition
    x = predictor.input_layer(sequence)
    x = self._layers(predictor.blocks, x, c, mask, local.expand(len(patch_lengths), -1))
    out = predictor.output_layer(x, c)[:, CHUNK * n :].unflatten(1, (n, CHUNK))
    return out[:, :, 1:][valid]
