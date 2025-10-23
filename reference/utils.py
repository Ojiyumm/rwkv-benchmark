########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import torch
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop

@MyStatic
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

@MyStatic # !!! will modify logits inplace !!!
def sampler_simple_batch(logits: torch.Tensor, noise: float = 0.0, temp: float = 1.0):
    assert temp > 0, "use noise=0 for greedy decoding"
    with torch.no_grad():
        if temp != 1.0:
            logits.mul_(1.0 / temp)
        if noise != 0.0:
            logits.add_(torch.empty_like(logits).uniform_(0.0, noise))
        return torch.argmax(logits, dim=-1, keepdim=True)

@MyStatic # !!! will modify logits inplace !!!
def sampler_simple(logits: torch.Tensor, noise: float = 0.0, temp: float = 1.0):
    assert temp > 0, "use noise=0 for greedy decoding"
    with torch.no_grad():
        if temp != 1.0:
            logits.mul_(1.0 / temp)
        if noise != 0.0:
            logits.add_(torch.empty_like(logits).uniform_(0.0, noise))
        return torch.argmax(logits, dim=-1, keepdim=False)

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        self.idx2token = {0:"<|endoftext|>".encode("utf-8")} # add <|endoftext|>
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            if k != 0: # ignore <|endoftext|>
                self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens, utf8_errors="strict"):
        return self.decodeBytes(tokens).decode('utf-8', errors=utf8_errors)

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()


@MyStatic
def sample_logits_batch(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    alpha_presence: float = 0.0,
    alpha_frequency: float = 0.0,
    token_counts: torch.Tensor = None
) -> torch.Tensor:
    """
    Advanced batch sampling with top_k, top_p, and repetition penalties

    Args:
        logits: [batch_size, vocab_size] logits tensor
        temperature: sampling temperature
        top_p: nucleus sampling probability
        top_k: top-k sampling limit
        alpha_presence: presence penalty (0.0 = disabled)
        alpha_frequency: frequency penalty (0.0 = disabled)
        token_counts: [batch_size, vocab_size] token frequency counts for penalties

    Returns:
        sampled_tokens: [batch_size, 1] tensor of sampled token indices
    """
    batch_size, vocab_size = logits.shape
    device = logits.device

    # Clone logits to avoid modifying input
    logits = logits.clone().float()

    # Apply repetition penalties if enabled
    if alpha_presence > 0.0 or alpha_frequency > 0.0:
        if token_counts is not None:
            # Presence penalty: -alpha_presence for any token that appeared
            if alpha_presence > 0.0:
                presence_mask = (token_counts > 0).float()
                logits -= alpha_presence * presence_mask

            # Frequency penalty: -alpha_frequency * count for each token
            if alpha_frequency > 0.0:
                logits -= alpha_frequency * token_counts.float()

    # Greedy path when temperature is effectively zero
    if temperature is not None and temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Apply top_k filtering
    if top_k > 0 and top_k < vocab_size:
        # Get top_k indices for each sample in batch
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

        # Create mask for top_k tokens
        top_k_mask = torch.zeros_like(probs, dtype=torch.bool, device=device)
        top_k_mask.scatter_(1, top_k_indices, True)

        # Zero out probabilities not in top_k
        probs = probs * top_k_mask.float()

    # Apply top_p (nucleus) filtering
    if top_p < 1.0:
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff for each sample
        cutoff_mask = cumulative_probs <= top_p

        # Include at least one token (the highest probability one)
        cutoff_mask[:, 0] = True

        # Zero out probabilities beyond cutoff
        sorted_probs = sorted_probs * cutoff_mask.float()

        # Scatter back to original positions
        probs.zero_()
        probs.scatter_(1, sorted_indices, sorted_probs)

    # Renormalize probabilities
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # Handle edge case of all-zero probabilities
    probs = torch.where(torch.sum(probs, dim=-1, keepdim=True) == 0,
                       torch.ones_like(probs) / vocab_size,
                       probs)

    # Sample from the distribution
    sampled_tokens = torch.multinomial(probs, num_samples=1)

    return sampled_tokens
