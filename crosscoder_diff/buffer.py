from utils import *
from transformer_lens import ActivationCache
import tqdm

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        try:
            d_model_A = model_A.cfg.d_model
            d_model_B = model_B.cfg.d_model
        except Exception as e:
            print(model_A.module.cfg)
            d_model_A = model_A.module.cfg.d_model
            d_model_B = model_B.module.cfg.d_model

        assert d_model_A == d_model_B
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.buffer = torch.zeros(
            (self.buffer_size, 2, d_model_A),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"]) # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
        [
            estimated_norm_scaling_factor_A,
            estimated_norm_scaling_factor_B,
        ],
        device="cuda:0",
        dtype=torch.float32,
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        print("Refreshing the buffer!")
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg["model_batch_size"], num_batches
                    )
                ]
                
                _, cache_A = self.model_A.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_A: ActivationCache

                _, cache_B = self.model_B.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_B: ActivationCache

                acts = torch.stack([cache_A[self.cfg["hook_point"]], cache_B[self.cfg["hook_point"]]], dim=0)
                acts = acts[:, :, 1:, :] # Drop BOS
                assert acts.shape == (2, tokens.shape[0], tokens.shape[1]-1, self.model_A.cfg.d_model) # [2, batch, seq_len, d_model]
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
