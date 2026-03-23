import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, matplotlib, random, math, time, os, copy, heapq
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum, auto
from collections import deque, defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}", end="")
if DEVICE.type == "cuda":
    print(f"  |  {torch.cuda.get_device_name(0)}"
          f"  |  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")
else:
    print()

  
#  ENUMS
  
class Sex(Enum):         MALE=0;   FEMALE=1
class Profession(Enum): FARMER=0; TRADER=1; WARRIOR=2; EXPLORER=3; HEALER=4
class Governance(Enum): DEMOCRACY=0; AUTOCRACY=1; THEOCRACY=2; ANARCHY=3
class TerrainType(Enum):PLAINS=0; FOREST=1; DESERT=2; MOUNTAIN=3; RIVER=4
class GoalType(Enum):   FORAGE=0; RETURN=1; FLEE=2; TRADE=3; EXPLORE=4; DEFEND=5
class DisasterType(Enum): DROUGHT=0; FLOOD=1; FIRE=2; PLAGUE=3

  
#  CONFIG
  
@dataclass
class SimConfig:
    # World
    world_w: int   = 240
    world_h: int   = 160
    terrain_tile:  int   = 16       # tiles per axis for terrain grid
    n_bad_zones: int       = 3
    bad_zone_radius: float = 8.0
    bad_zone_damage: float = 0.4
    fog_radius: float      = 18.0   # visibility radius per organism

    # Seasons
    season_period: int   = 1000
    season_amp:    float = 0.40

    # Disasters
    disaster_prob:  float = 0.0008  # per step base probability
    disaster_duration: int= 120

    # Villages / Civilization
    init_villages: int       = 2
    max_villages:  int       = 12
    village_radius: float    = 16.0
    village_food_radius: float = 2.8
    clan_form_density: int   = 18
    clan_form_radius:  float = 24.0
    clan_form_cooldown:int   = 250
    village_breach_radius: float    = 30.0
    village_breach_kill_prob: float = 0.20
    min_warriors_per_village: int   = 3
    warrior_respawn_interval: int   = 120
    territory_radius: float         = 40.0
    overpop_threshold: int          = 50

    # Resources
    resource_types: int = 4    # food, wood, metal, medicine
    trade_reward:   float= 2.0

    # Communication
    msg_dim: int = 8

    # Populations
    init_organisms: int  = 110
    init_warriors:  int  = 24
    init_enemies:   int  = 6
    max_organisms:  int  = 320
    max_warriors:   int  = 100
    max_enemies:    int  = 45

    # Food / Resources
    init_food: int         = 350
    max_food:  int         = 550
    food_spawn_rate: float = 0.65
    food_eat_radius: float = 2.5

    # Organism physics
    org_friction: float = 0.76
    org_accel:    float = 0.94

    # Energy
    init_energy: float         = 108.0
    max_energy:  float         = 200.0
    base_metabolism: float     = 0.026
    move_cost_factor: float    = 0.009
    food_energy: float         = 75.0
    reproduce_threshold: float = 80.0
    reproduce_cost:      float = 24.0
    gestation_steps: int       = 12
    starve_energy: float       = 0.0
    elder_age: int             = 450
    elder_birth_bonus: float   = 0.28
    sick_duration: int         = 70
    sick_damage: float         = 0.07

    # Genes
    gene_vision_range:    Tuple = (5.0, 26.0)
    gene_meta_range:      Tuple = (0.02, 0.22)
    gene_fertility_range: Tuple = (0.3, 1.0)
    gene_va_range:        Tuple = (0.0, 1.0)
    gene_speed_range_M:   Tuple = (1.5, 4.5)
    gene_aggro_range_M:   Tuple = (0.2, 1.0)
    gene_speed_range_F:   Tuple = (0.8, 3.0)
    gene_aggro_range_F:   Tuple = (0.0, 0.6)
    mutation_rate:   float = 0.10
    mutation_strength: float = 0.11
    speciation_threshold: float = 1.8  # gene distance for new species

    # Warrior
    warrior_max_speed:    float = 4.2
    warrior_accel:        float = 1.2
    warrior_friction:     float = 0.79
    warrior_vision:       float = 24.0
    warrior_energy_init:  float = 145.0
    warrior_energy_max:   float = 250.0
    warrior_metabolism:   float = 0.042
    warrior_move_cost:    float = 0.008
    warrior_kill_radius:  float = 4.2
    warrior_protect_r:    float = 20.0
    warrior_food_energy:  float = 42.0
    warrior_rep_threshold:float = 175.0
    warrior_rep_cost:     float = 50.0
    warrior_attack_bonus: float = 32.0
    war_chief_bonus:      float = 0.28

    # Enemy
    enemy_max_speed_init:  float = 1.9
    enemy_accel:           float = 0.85
    enemy_friction:        float = 0.73
    enemy_kill_radius_init:float = 1.9
    enemy_vision_init:     float = 14.0
    enemy_max_speed_cap:   float = 5.0
    enemy_kill_radius_cap: float = 4.5
    enemy_growth_interval: int   = 500
    enemy_growth_rate:     float = 0.055
    enemy_spawn_interval:  int   = 800
    enemy_pack_radius:     float = 30.0

   
    n_goals:         int   = 6     # number of high-level goals
    goal_horizon:    int   = 20    # steps per high-level decision
    goal_embed_dim:  int   = 16    # goal embedding size

   
    wm_latent_dim:   int   = 32    # world model latent state
    wm_rollout:      int   = 3     # dream steps before acting
    wm_train_freq:   int   = 500   # train world model every N steps
    wm_lr:           float = 3e-4

    
    mem_len:         int   = 8     # memory window
    mem_heads:       int   = 4     # attention heads
    mem_dim:         int   = 64    # memory embedding dim

   
    neat_pop_size:   int   = 8     # network variants in evolution pool
    neat_mutate_freq:int   = 600   # mutate network structure every N steps
    neat_elites:     int   = 2     # top k networks survive

    
    act_dim:          int   = 9
    org_obs_dim:      int   = 56    # expanded observation
    warrior_obs_dim:  int   = 38
    global_state_dim: int   = 32
    hidden_dim:       int   = 256
    gru_hidden:       int   = 128
    lr:               float = 1.2e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.2
    entropy_coef:     float = 0.018
    value_coef:       float = 0.5
    ppo_epochs:       int   = 4
    batch_size:       int   = 1024
    rollout_len:      int   = 320
    max_grad_norm:    float = 0.5
    pbt_population:   int   = 4
    pbt_interval:     int   = 1200
    curiosity_coef:   float = 0.025
    curiosity_grid_w: int   = 24
    curiosity_grid_h: int   = 16


    curriculum_target_survival: float = 0.35  # target fraction surviving
    curriculum_check_interval:  int   = 300

   
    selfplay_archive_size: int  = 5
    selfplay_swap_interval:int  = 2000

    # Simulation
    max_steps:    int  = 8000
    render_every: int  = 100
    save_plot:    bool = True
    plot_dir:     str  = "sim_frames_v6"

CFG = SimConfig()

_ACT_DIRS = np.array([
    [ 0,-1],[ 1,-1],[ 1, 0],[ 1, 1],
    [ 0, 1],[-1, 1],[-1, 0],[-1,-1],[ 0, 0],
], dtype=np.float32)
for _i in range(8):
    _n = np.linalg.norm(_ACT_DIRS[_i])
    if _n > 0: _ACT_DIRS[_i] /= _n

TERRAIN_COLORS = {
    TerrainType.PLAINS:   "#2d4a1e",
    TerrainType.FOREST:   "#1a3a12",
    TerrainType.DESERT:   "#5a4a1a",
    TerrainType.MOUNTAIN: "#3a3a3a",
    TerrainType.RIVER:    "#1a2a4a",
}
TERRAIN_FOOD_MULT = {
    TerrainType.PLAINS:   1.0,
    TerrainType.FOREST:   1.5,
    TerrainType.DESERT:   0.3,
    TerrainType.MOUNTAIN: 0.5,
    TerrainType.RIVER:    1.3,
}
TERRAIN_SPEED_MULT = {
    TerrainType.PLAINS:   1.0,
    TerrainType.FOREST:   0.8,
    TerrainType.DESERT:   0.7,
    TerrainType.MOUNTAIN: 0.5,
    TerrainType.RIVER:    0.9,
}

  
#  TERRAIN MAP
  
class TerrainMap:
    def __init__(self):
        W, H = CFG.terrain_tile, int(CFG.terrain_tile * CFG.world_h / CFG.world_w)
        self.W = W; self.H = max(H, 1)
        self.grid = np.full((self.H, W), TerrainType.PLAINS)
        self._generate()

    def _generate(self):
        # Simple noise-based terrain generation
        for ty in range(self.H):
            for tx in range(self.W):
                x = tx / self.W; y = ty / self.H
                # Rivers: diagonal bands
                if abs(math.sin(x * math.pi * 3 + 0.5) * math.cos(y * math.pi * 2) - 0.05) < 0.08:
                    self.grid[ty, tx] = TerrainType.RIVER
                # Mountains: corners
                elif (x < 0.15 or x > 0.85) and (y < 0.2 or y > 0.8):
                    self.grid[ty, tx] = TerrainType.MOUNTAIN
                # Desert: centre-top band
                elif 0.3 < x < 0.7 and 0.05 < y < 0.25:
                    self.grid[ty, tx] = TerrainType.DESERT
                # Forest: edges
                elif x < 0.2 or x > 0.8 or y < 0.15 or y > 0.85:
                    self.grid[ty, tx] = TerrainType.FOREST

    def get(self, x: float, y: float) -> TerrainType:
        tx = int(np.clip(x / CFG.world_w * self.W, 0, self.W - 1))
        ty = int(np.clip(y / CFG.world_h * self.H, 0, self.H - 1))
        return self.grid[ty, tx]

    def food_mult(self, x, y): return TERRAIN_FOOD_MULT[self.get(x, y)]
    def speed_mult(self, x, y): return TERRAIN_SPEED_MULT[self.get(x, y)]

TERRAIN = TerrainMap()

  
#  GENOME  (expanded with cultural + speciation genes)
  
@dataclass

