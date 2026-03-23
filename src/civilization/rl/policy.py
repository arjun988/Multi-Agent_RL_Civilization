from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..core.config import CFG, DEVICE, Sex


@dataclass
class Genome:
    speed: float=2.0;  vision: float=11.0; metabolism: float=0.10
    aggression: float=0.4; fertility: float=0.6
    village_affinity: float=0.5; dominance: float=0.5
    immune_strength: float=0.5; trade_drive: float=0.5
    cooperation: float=0.5      # tendency to help nearby kin
    curiosity_drive: float=0.5  # intrinsic exploration tendency
    cultural_memory: float=0.5  # how much offspring inherit culture
    species_id: int = 0         # set during speciation

    @classmethod
    def random(cls, sex: Sex) -> "Genome":
        sr = CFG.gene_speed_range_M if sex==Sex.MALE else CFG.gene_speed_range_F
        ar = CFG.gene_aggro_range_M if sex==Sex.MALE else CFG.gene_aggro_range_F
        return cls(speed=random.uniform(*sr),
                   vision=random.uniform(*CFG.gene_vision_range),
                   metabolism=random.uniform(*CFG.gene_meta_range),
                   aggression=random.uniform(*ar),
                   fertility=random.uniform(*CFG.gene_fertility_range),
                   village_affinity=random.uniform(*CFG.gene_va_range),
                   dominance=random.uniform(0.2, 0.8),
                   immune_strength=random.uniform(0.1, 0.9),
                   trade_drive=random.uniform(0.0, 1.0),
                   cooperation=random.uniform(0.0, 1.0),
                   curiosity_drive=random.uniform(0.0, 1.0),
                   cultural_memory=random.uniform(0.2, 0.8))

    def mutate(self, sex: Sex) -> "Genome":
        sr = CFG.gene_speed_range_M if sex==Sex.MALE else CFG.gene_speed_range_F
        ar = CFG.gene_aggro_range_M if sex==Sex.MALE else CFG.gene_aggro_range_F
        def m(v, lo, hi):
            if random.random() < CFG.mutation_rate:
                v += random.gauss(0, CFG.mutation_strength) * (hi - lo)
            return float(np.clip(v, lo, hi))
        return Genome(speed=m(self.speed,*sr),
                      vision=m(self.vision,*CFG.gene_vision_range),
                      metabolism=m(self.metabolism,*CFG.gene_meta_range),
                      aggression=m(self.aggression,*ar),
                      fertility=m(self.fertility,*CFG.gene_fertility_range),
                      village_affinity=m(self.village_affinity,*CFG.gene_va_range),
                      dominance=m(self.dominance,0.1,0.9),
                      immune_strength=m(self.immune_strength,0.0,1.0),
                      trade_drive=m(self.trade_drive,0.0,1.0),
                      cooperation=m(self.cooperation,0.0,1.0),
                      curiosity_drive=m(self.curiosity_drive,0.0,1.0),
                      cultural_memory=m(self.cultural_memory,0.1,0.9),
                      species_id=self.species_id)

    def crossover(self, other: "Genome", csex: Sex) -> "Genome":
        dm = other.dominance; df = self.dominance
        total = dm + df + 1e-8; pf = dm / total
        pick = lambda a, b: b if random.random() < pf else a
        return Genome(speed=pick(self.speed,other.speed),
                      vision=pick(self.vision,other.vision),
                      metabolism=pick(self.metabolism,other.metabolism),
                      aggression=pick(self.aggression,other.aggression),
                      fertility=pick(self.fertility,other.fertility),
                      village_affinity=pick(self.village_affinity,other.village_affinity),
                      dominance=(self.dominance+other.dominance)/2,
                      immune_strength=pick(self.immune_strength,other.immune_strength),
                      trade_drive=pick(self.trade_drive,other.trade_drive),
                      cooperation=pick(self.cooperation,other.cooperation),
                      curiosity_drive=pick(self.curiosity_drive,other.curiosity_drive),
                      cultural_memory=pick(self.cultural_memory,other.cultural_memory),
                      species_id=self.species_id).mutate(csex)

    def distance(self, other: "Genome") -> float:
        """Genetic distance for speciation."""
        return math.sqrt(sum((a-b)**2 for a,b in [
            (self.speed/4.5,        other.speed/4.5),
            (self.vision/26.,       other.vision/26.),
            (self.aggression,       other.aggression),
            (self.cooperation,      other.cooperation),
            (self.trade_drive,      other.trade_drive),
            (self.curiosity_drive,  other.curiosity_drive),
        ]))

    def to_vec(self) -> np.ndarray:
        def norm(v, lo, hi): return (v-lo)/(hi-lo+1e-8)
        return np.array([
            norm(self.speed,1.0,4.5), norm(self.vision,*CFG.gene_vision_range),
            norm(self.metabolism,*CFG.gene_meta_range), norm(self.aggression,0.,1.),
            norm(self.fertility,*CFG.gene_fertility_range),
            self.village_affinity, self.dominance, self.immune_strength,
            self.trade_drive, self.cooperation, self.curiosity_drive, self.cultural_memory,
        ], dtype=np.float32)  # 12 genes

  
#  TRANSFORMER MEMORY MODULE
  
class TransformerMemory(nn.Module):
    """
    Replaces GRU with a small causal transformer memory window.
    Each agent maintains a rolling buffer of past mem_len embeddings.
    Multi-head self-attention reads across the window to produce context.
    """
    def __init__(self, input_dim: int, mem_dim: int = 64,
                 heads: int = 4, mem_len: int = 8):
        super().__init__()
        self.mem_len = mem_len
        self.mem_dim = mem_dim
        self.proj = nn.Linear(input_dim, mem_dim)
        self.attn = nn.MultiheadAttention(mem_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(mem_dim)
        self.out  = nn.Linear(mem_dim, mem_dim)
        # Positional encoding
        pe = torch.zeros(mem_len, mem_dim)
        pos = torch.arange(mem_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, mem_dim, 2).float() * (-math.log(10000.0)/mem_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:mem_dim//2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, mem_buf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:       (B, input_dim)  current observation
        mem_buf: (B, mem_len, mem_dim)  past memory
        Returns: context (B, mem_dim), new_mem_buf
        """
        emb = self.proj(x).unsqueeze(1)  # (B, 1, mem_dim)
        new_buf = torch.cat([mem_buf[:, 1:, :], emb], dim=1)  # roll window
        buf_pe  = new_buf + self.pe.unsqueeze(0)
        ctx, _  = self.attn(buf_pe, buf_pe, buf_pe)
        ctx     = self.norm(ctx[:, -1, :])  # use most recent token
        ctx     = self.out(ctx)
        return ctx, new_buf

  
#  WORLD MODEL  (Dreamer-lite: predict next latent + reward)
  
class WorldModel(nn.Module):
    """
    Learns to predict: next_obs, reward given (obs, action).
    Agents use this to mentally simulate CFG.wm_rollout steps
    before committing to an action.
    """
    def __init__(self, obs_dim: int, act_dim: int, latent: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, latent)
        )
        self.trans = nn.Sequential(
            nn.Linear(latent + act_dim, 128), nn.Tanh(),
            nn.Linear(128, latent)
        )
        self.dec_obs    = nn.Linear(latent, obs_dim)
        self.dec_reward = nn.Linear(latent, 1)
        self._buf: List = []

    def forward(self, obs, act_onehot):
        z    = self.enc(obs)
        z2   = self.trans(torch.cat([z, act_onehot], -1))
        pred_obs = self.dec_obs(z2)
        pred_rew = self.dec_reward(z2)
        return pred_obs, pred_rew, z2

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.enc(obs)

    def dream_reward(self, obs_np: np.ndarray, act: int, steps: int = 3) -> float:
        """Quick dreaming: simulate steps forward, sum predicted rewards."""
        if len(self._buf) < 64: return 0.0
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total = 0.0
        for _ in range(steps):
            ah = torch.zeros(1, CFG.act_dim, device=DEVICE)
            ah[0, act] = 1.0
            with torch.no_grad():
                obs_t, rew, _ = self(obs_t, ah)
            total += rew.item()
        return float(total)

    def store(self, obs, act, next_obs, rew):
        self._buf.append((obs, act, next_obs, rew))
        if len(self._buf) > 20000: self._buf.pop(0)

    def train_step(self, opt) -> float:
        if len(self._buf) < 128: return 0.0
        batch = random.sample(self._buf, min(256, len(self._buf)))
        obs   = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=DEVICE)
        acts  = torch.tensor([b[1] for b in batch], dtype=torch.long,    device=DEVICE)
        nobs  = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        rews  = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        ah    = F.one_hot(acts, num_classes=CFG.act_dim).float()
        p_obs, p_rew, _ = self(obs, ah)
        loss = F.mse_loss(p_obs, nobs) + F.mse_loss(p_rew, rews)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        return loss.item()

  
#  HRL   High-Level Goal Policy + Low-Level Goal-Conditioned Actor
  
class HighLevelPolicy(nn.Module):
    """Outputs a goal index every goal_horizon steps."""
    def __init__(self, obs_dim: int, n_goals: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, n_goals)
        )
        self.goal_emb = nn.Embedding(n_goals, CFG.goal_embed_dim)

    def forward(self, obs: torch.Tensor):
        return self.net(obs)

    def embed(self, goal_idx: int) -> torch.Tensor:
        return self.goal_emb(torch.tensor(goal_idx, device=DEVICE))


class HRLWrapper:
    """Wraps a goal-conditioned low-level policy with a high-level scheduler."""
    def __init__(self, obs_dim: int):
        self.hl = HighLevelPolicy(obs_dim, CFG.n_goals).to(DEVICE)
        self.hl_opt = optim.Adam(self.hl.parameters(), lr=5e-4)
        self.goal_embed_dim = CFG.goal_embed_dim
        # Per-agent current goal
        self.agent_goals: Dict[int, int] = {}
        self.agent_goal_steps: Dict[int, int] = {}

    def get_goal(self, agent_id: int, obs_np: np.ndarray) -> Tuple[int, np.ndarray]:
        steps = self.agent_goal_steps.get(agent_id, 0)
        if steps <= 0 or agent_id not in self.agent_goals:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = self.hl(obs_t)
            goal = torch.distributions.Categorical(logits=logits).sample().item()
            self.agent_goals[agent_id] = goal
            self.agent_goal_steps[agent_id] = CFG.goal_horizon
        else:
            goal = self.agent_goals[agent_id]
        self.agent_goal_steps[agent_id] -= 1
        with torch.no_grad():
            embed = self.hl.embed(goal).cpu().numpy()
        return goal, embed

    def clear(self, agent_id: int):
        self.agent_goals.pop(agent_id, None)
        self.agent_goal_steps.pop(agent_id, None)

  
#  NEAT-STYLE NEURAL EVOLUTION POOL
  
class NEATPool:
    """
    Maintains a small population of network variants.
    Every neat_mutate_freq steps, prunes worst, mutates best.
    """
    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = CFG.act_dim
        self.population: List[nn.Module] = []
        self.scores: List[float] = []
        for _ in range(CFG.neat_pop_size):
            net = self._make_net()
            self.population.append(net)
            self.scores.append(0.0)
        self.active_idx = 0

    def _make_net(self) -> nn.Module:
        hidden = random.choice([64, 128, 192, 256])
        layers = random.choice([2, 3])
        mods = []
        inp = self.obs_dim + CFG.goal_embed_dim
        for _ in range(layers):
            mods += [nn.Linear(inp, hidden), nn.Tanh()]
            inp = hidden
        mods.append(nn.Linear(hidden, self.act_dim))
        net = nn.Sequential(*mods).to(DEVICE)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        return net

    def get_active(self) -> nn.Module:
        return self.population[self.active_idx]

    def update_score(self, reward: float):
        self.scores[self.active_idx] = self.scores[self.active_idx]*0.95 + reward*0.05

    def evolve(self):
        """Prune worst, mutate best."""
        order = sorted(range(len(self.scores)), key=lambda i: self.scores[i])
        # Replace bottom half with mutated copies of top
        for i in order[:len(order)//2]:
            src_idx = order[-1]
            src = self.population[src_idx]
            child = copy.deepcopy(src)
            with torch.no_grad():
                for p in child.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            self.population[i] = child
            self.scores[i] = 0.0
        # Best network stays active
        self.active_idx = order[-1]
        print(f"  [NEAT] Evolved. Best score={self.scores[self.active_idx]:.3f}")

  
#  COMMUNICATION  (learned message passing)
  
class CommModule(nn.Module):
    """
    Agents broadcast msg_dim message vectors.
    Nearby agents aggregate received messages.
    """
    def __init__(self, obs_dim: int, msg_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, msg_dim), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(msg_dim, 32), nn.Tanh(),
            nn.Linear(32, msg_dim)
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def aggregate(self, messages: List[torch.Tensor]) -> torch.Tensor:
        if not messages:
            return torch.zeros(CFG.msg_dim, device=DEVICE)
        flat = [m.squeeze(0) if m.dim() == 2 else m for m in messages]
        stacked = torch.stack(flat, dim=0)   # (K, msg_dim)
        return self.decoder(stacked.mean(0)) # (msg_dim,)

  
#  MAIN ACTOR-CRITIC  (MAPPO + Transformer Memory + Goal-conditioned)
  
class CivActorCritic(nn.Module):
    """
    Full architecture:
    obs  transformer_memory  [mem_ctx, goal_embed, msg]  GRU  actor/critic
    """
    def __init__(self, obs_dim: int, act_dim: int, global_dim: int):
        super().__init__()
        self.tmem = TransformerMemory(obs_dim, CFG.mem_dim, CFG.mem_heads, CFG.mem_len)
        gru_in = CFG.mem_dim + CFG.goal_embed_dim + CFG.msg_dim
        self.gru = nn.GRUCell(gru_in, CFG.gru_hidden)
        self.actor = nn.Sequential(
            nn.Linear(CFG.gru_hidden, CFG.hidden_dim//2), nn.Tanh(),
            nn.Linear(CFG.hidden_dim//2, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(global_dim, CFG.hidden_dim), nn.LayerNorm(CFG.hidden_dim), nn.Tanh(),
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim//2), nn.Tanh(),
            nn.Linear(CFG.hidden_dim//2, 1)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def actor_step(self, obs_t, mem_buf, hx, goal_embed, msg_embed):
        ctx, new_buf = self.tmem(obs_t, mem_buf)
        inp  = torch.cat([ctx, goal_embed, msg_embed], dim=-1)
        hx   = self.gru(inp, hx)
        return self.actor(hx), hx, new_buf

    def critic_step(self, gs_t):
        return self.critic(gs_t)

  
#  PBT CONFIG
  
@dataclass
class PBTConfig:
    lr: float=1.2e-4; entropy_coef: float=0.018
    gamma: float=0.99; clip_eps: float=0.2; score: float=0.0

    def mutate(self) -> "PBTConfig":
        c = copy.copy(self)
        c.lr           = float(np.clip(c.lr*random.uniform(0.8,1.25), 1e-5, 5e-3))
        c.entropy_coef = float(np.clip(c.entropy_coef*random.uniform(0.8,1.25), 0.001, 0.1))
        c.gamma        = float(np.clip(c.gamma+random.uniform(-0.01,0.01), 0.90, 0.999))
        c.clip_eps     = float(np.clip(c.clip_eps*random.uniform(0.85,1.15), 0.05, 0.5))
        c.score        = 0.0; return c

  
#  CIV POLICY  (MAPPO + Transformer + HRL + WorldModel + PBT + NEAT)
  
class CivPolicy:
    def __init__(self, obs_dim: int, global_dim: int, name: str = "policy"):
        self.name = name
        self.obs_dim = obs_dim
        self.pbt_configs = [PBTConfig() for _ in range(CFG.pbt_population)]
        self.active_pbt  = 0
        self.net = CivActorCritic(obs_dim, CFG.act_dim, global_dim).to(DEVICE)
        self.comm = CommModule(obs_dim, CFG.msg_dim).to(DEVICE)
        self.world_model = WorldModel(obs_dim, CFG.act_dim, CFG.wm_latent_dim).to(DEVICE)
        self.wm_opt = optim.Adam(self.world_model.parameters(), lr=CFG.wm_lr)
        self.hrl = HRLWrapper(obs_dim)
        self.neat = NEATPool(obs_dim)
        self._make_opt()
        # Per-agent states
        self.hidden:  Dict[int, torch.Tensor] = {}
        self.mem_buf: Dict[int, torch.Tensor] = {}
        self.msgs:    Dict[int, torch.Tensor] = {}  # last broadcast messages
        self._reset()
        self.visit_grid = np.zeros((CFG.curiosity_grid_h, CFG.curiosity_grid_w), dtype=np.float32)
        self.steps = 0
        self.wm_loss_hist: List[float] = []

        # Self-play archive: store copies of past nets
        self.archive: List[nn.Module] = []

        # Auto-curriculum state
        self.survival_history: deque = deque(maxlen=100)
        self.curriculum_difficulty = 1.0  # multiplier on enemy strength

    def _make_opt(self):
        cfg = self.pbt_configs[self.active_pbt]
        params = list(self.net.parameters()) + list(self.comm.parameters())
        self.opt = optim.Adam(params, lr=cfg.lr, eps=1e-5)

    def _reset(self):
        self.buf = {k: [] for k in ["obs","acts","logps","vals","rews","dones","hx","gs","goal_e","msg_e"]}

    def _get_hidden(self, aid: int) -> torch.Tensor:
        if aid not in self.hidden:
            self.hidden[aid] = torch.zeros(1, CFG.gru_hidden, device=DEVICE)
        return self.hidden[aid]

    def _get_membuf(self, aid: int) -> torch.Tensor:
        if aid not in self.mem_buf:
            self.mem_buf[aid] = torch.zeros(1, CFG.mem_len, CFG.mem_dim, device=DEVICE)
        return self.mem_buf[aid]

    def _get_msg(self, aid: int) -> torch.Tensor:
        if aid not in self.msgs:
            self.msgs[aid] = torch.zeros(1, CFG.msg_dim, device=DEVICE)
        return self.msgs[aid]

    def clear_agent(self, aid: int):
        for d in [self.hidden, self.mem_buf, self.msgs]:
            d.pop(aid, None)
        self.hrl.clear(aid)

    def intrinsic_reward(self, x, y) -> float:
        gx = int(np.clip(x/CFG.world_w*CFG.curiosity_grid_w, 0, CFG.curiosity_grid_w-1))
        gy = int(np.clip(y/CFG.world_h*CFG.curiosity_grid_h, 0, CFG.curiosity_grid_h-1))
        cnt = self.visit_grid[gy, gx]; self.visit_grid[gy, gx] += 1.0
        return CFG.curiosity_coef / (1.0 + math.sqrt(cnt))

    def store(self, obs, act, logp, val, rew, done, hx, gs, ge, me):
        self.buf["obs"].append(obs);   self.buf["acts"].append(act)
        self.buf["logps"].append(logp);self.buf["vals"].append(val)
        self.buf["rews"].append(rew);  self.buf["dones"].append(float(done))
        self.buf["hx"].append(hx.cpu().numpy().flatten())
        self.buf["gs"].append(gs); self.buf["goal_e"].append(ge)
        self.buf["msg_e"].append(me)

    @torch.no_grad()
    def batch_act(self, obs_list, agent_ids, global_states,
                  nearby_agent_ids_list):
        if not obs_list: return [], [], [], None, [], []
        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE)
        gs_t  = torch.tensor(np.array(global_states), dtype=torch.float32, device=DEVICE)
        N = len(obs_list)

        # Compute broadcast messages for each agent
        msgs_t = self.comm.encode(obs_t)   # (N, msg_dim)
        for i, aid in enumerate(agent_ids):
            self.msgs[aid] = msgs_t[i:i+1].detach()

        # Aggregate incoming messages per agent  each agg is (msg_dim,)
        agg_msgs = []
        for i, (aid, nearby_ids) in enumerate(zip(agent_ids, nearby_agent_ids_list)):
            incoming = [self.msgs[nid] for nid in nearby_ids if nid in self.msgs and nid != aid]
            agg = self.comm.aggregate(incoming)          # (msg_dim,)
            agg_msgs.append(agg)
        agg_msg_t = torch.stack(agg_msgs, dim=0).to(DEVICE)   # (N, msg_dim) 

        # HRL: get goal embeddings
        goal_embeds = []
        goal_indices = []
        for i, (aid, obs_np) in enumerate(zip(agent_ids, obs_list)):
            goal_idx, goal_e = self.hrl.get_goal(aid, obs_np)
            goal_indices.append(goal_idx)
            goal_embeds.append(goal_e)
        goal_t = torch.tensor(np.array(goal_embeds), dtype=torch.float32, device=DEVICE)

        # Stack per-agent hidden + mem_buf
        hx_list  = [self._get_hidden(aid) for aid in agent_ids]
        mb_list  = [self._get_membuf(aid) for aid in agent_ids]
        hx_t     = torch.cat(hx_list,  dim=0)           # (N, gru_h)
        mb_t     = torch.cat(mb_list,  dim=0)           # (N, mem_len, mem_dim)

        # Forward pass
        ctx, new_mb = self.net.tmem(obs_t, mb_t)
        inp  = torch.cat([ctx, goal_t, agg_msg_t], dim=-1)
        hx_new = self.net.gru(inp, hx_t)
        logits = self.net.actor(hx_new)
        vals   = self.net.critic_step(gs_t)

        # NEAT net also votes (blend)
        neat_net = self.neat.get_active()
        neat_inp = torch.cat([obs_t, goal_t], dim=-1)
        neat_logits = neat_net(neat_inp)
        logits = 0.7 * logits + 0.3 * neat_logits  # ensemble

        dist   = torch.distributions.Categorical(logits=logits)
        acts   = dist.sample(); logps = dist.log_prob(acts)

        # Update per-agent states
        for i, aid in enumerate(agent_ids):
            self.hidden[aid]  = hx_new[i:i+1].detach()
            self.mem_buf[aid] = new_mb[i:i+1].detach()

        return (acts.cpu().numpy(), logps.cpu().numpy(),
                vals.squeeze(-1).cpu().numpy(),
                hx_new.detach(),
                [goal_embeds[i] for i in range(N)],
                [agg_msgs[i].detach().cpu().numpy() for i in range(N)])

    def update(self, global_score: float = 0.0, survival_frac: float = 1.0) -> dict:
        n = len(self.buf["obs"])
        if n < CFG.batch_size: return {}
        cfg = self.pbt_configs[self.active_pbt]
        cfg.score = cfg.score * 0.95 + global_score * 0.05
        self.survival_history.append(survival_frac)

        obs   = torch.tensor(np.array(self.buf["obs"]),   dtype=torch.float32, device=DEVICE)
        acts  = torch.tensor(self.buf["acts"],             dtype=torch.long,    device=DEVICE)
        logps = torch.tensor(self.buf["logps"],            dtype=torch.float32, device=DEVICE)
        vals  = torch.tensor(self.buf["vals"],             dtype=torch.float32, device=DEVICE)
        rews  = torch.tensor(self.buf["rews"],             dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(self.buf["dones"],            dtype=torch.float32, device=DEVICE)
        hxs   = torch.tensor(np.array(self.buf["hx"]),    dtype=torch.float32, device=DEVICE)
        hxs   = hxs.view(n, 1, CFG.gru_hidden)
        gs    = torch.tensor(np.array(self.buf["gs"]),    dtype=torch.float32, device=DEVICE)
        ge    = torch.tensor(np.array(self.buf["goal_e"]),dtype=torch.float32, device=DEVICE)
        me    = torch.tensor(np.array(self.buf["msg_e"]), dtype=torch.float32, device=DEVICE)

        adv = torch.zeros_like(rews); gae = 0.0
        for t in reversed(range(n)):
            nv = 0.0 if t==n-1 or dones[t] else vals[t+1].item()
            delta = rews[t] + cfg.gamma*nv*(1-dones[t]) - vals[t]
            gae   = delta + cfg.gamma*CFG.gae_lambda*(1-dones[t])*gae
            adv[t]= gae
        ret = adv + vals
        adv = (adv-adv.mean())/(adv.std()+1e-8)

        m = {"pl":0,"vl":0,"ent":0,"n":0}
        idx = torch.randperm(n)
        for _ in range(CFG.ppo_epochs):
            for s in range(0, n, CFG.batch_size):
                mb   = idx[s:s+CFG.batch_size]
                hx_mb= hxs[mb].squeeze(1)
                # Dummy mem_buf for update (use zeros  approximate)
                zero_mb = torch.zeros(len(mb), CFG.mem_len, CFG.mem_dim, device=DEVICE)
                ctx_mb, _ = self.net.tmem(obs[mb], zero_mb)
                inp_mb    = torch.cat([ctx_mb, ge[mb], me[mb]], dim=-1)
                hx_new_mb = self.net.gru(inp_mb, hx_mb)
                logit2    = self.net.actor(hx_new_mb)
                v2        = self.net.critic_step(gs[mb])
                # NEAT blend
                neat_inp2 = torch.cat([obs[mb], ge[mb]], dim=-1)
                neat_l2   = self.neat.get_active()(neat_inp2)
                logit2    = 0.7*logit2 + 0.3*neat_l2
                d2   = torch.distributions.Categorical(logits=logit2)
                nlp  = d2.log_prob(acts[mb]); ent = d2.entropy().mean()
                r    = (nlp-logps[mb]).exp(); a = adv[mb]
                pl   = -torch.min(r*a, r.clamp(1-cfg.clip_eps,1+cfg.clip_eps)*a).mean()
                vl   = F.mse_loss(v2.squeeze(-1), ret[mb])
                loss = pl + CFG.value_coef*vl - cfg.entropy_coef*ent
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters())+list(self.comm.parameters()),
                    CFG.max_grad_norm)
                self.opt.step()
                m["pl"]+=pl.item(); m["vl"]+=vl.item(); m["ent"]+=ent.item(); m["n"]+=1

        self.neat.update_score(global_score)
        self._reset(); self.steps += n
        nu = max(m["n"], 1)
        return {k: v/nu for k,v in m.items() if k!="n"}

    def pbt_step(self):
        scores  = [c.score for c in self.pbt_configs]
        best_i  = int(np.argmax(scores)); worst_i = int(np.argmin(scores))
        if best_i!=worst_i and scores[best_i]>scores[worst_i]+0.5:
            self.pbt_configs[worst_i] = self.pbt_configs[best_i].mutate()
            print(f"  [PBT] {self.name}: replaced C{worst_i} with mut(C{best_i}) score={scores[best_i]:.2f}")
        self.active_pbt = best_i; self._make_opt()

    def neat_step(self):
        self.neat.evolve()

    def train_world_model(self):
        loss = self.world_model.train_step(self.wm_opt)
        if loss > 0: self.wm_loss_hist.append(loss)

    def auto_curriculum(self) -> float:
        """Adjust difficulty based on recent survival fraction."""
        if len(self.survival_history) < 20: return self.curriculum_difficulty
        avg = float(np.mean(self.survival_history))
        if avg < CFG.curriculum_target_survival - 0.1:
            # Too hard  reduce enemy aggression
            self.curriculum_difficulty = max(0.5, self.curriculum_difficulty - 0.02)
        elif avg > CFG.curriculum_target_survival + 0.1:
            # Too easy  increase
            self.curriculum_difficulty = min(2.0, self.curriculum_difficulty + 0.02)
        return self.curriculum_difficulty

    def archive_policy(self):
        if len(self.archive) >= CFG.selfplay_archive_size:
            self.archive.pop(0)
        self.archive.append(copy.deepcopy(self.net))
        print(f"  [Archive] {self.name}: saved policy (archive size={len(self.archive)})")

  
#  VILLAGE  (governance, multi-resource, culture)
  
_VID = 0



