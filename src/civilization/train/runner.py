from __future__ import annotations

import math
import os
import random
import time

import numpy as np

from ..core.config import CFG, Governance, Profession, Sex
from ..core.entities import Enemy, Organism
from ..env.world import World
from ..rl.policy import CivPolicy
from ..viz.renderer import Renderer, op_ref
def run():
    print("\n"+""*78)
    print("  ORGANISM LIVING SYSTEM   FULL CIVILIZATION AI")
    print("  HRL + WorldModel(Dreamer) + TransformerMemory + NEAT + PBT + Comm + Speciation")
    print("  Terrain + Disasters + Governance + Professions + Multi-Resource + Self-Play")
    print(""*78)
    print(f"  World {CFG.world_w}{CFG.world_h}  |  {CFG.terrain_tile}{CFG.terrain_tile} terrain tiles")
    print(f"  RL stack: MAPPO+GRU{CFG.gru_hidden}+Transformer(L={CFG.mem_len},H={CFG.mem_heads})")
    print(f"          + HRL({CFG.n_goals} goals, {CFG.goal_horizon}s) + WorldModel({CFG.wm_latent_dim}d)")
    print(f"          + NEAT(pop={CFG.neat_pop_size}) + PBT{CFG.pbt_population}")
    print(f"          + Communication({CFG.msg_dim}d) + AutoCurriculum + SelfPlay")
    print(""*78+"\n")

    world=World()
    op=CivPolicy(CFG.org_obs_dim,    CFG.global_state_dim, name="org")
    wp=CivPolicy(CFG.warrior_obs_dim,CFG.global_state_dim, name="warrior")
    op_ref.append(op)
    rend=Renderer(); t0=time.time(); mo={}; mw={}; respawns=0

    for step in range(1,CFG.max_steps+1):
        info=world.step(op,wp)

        pop_before = sum(world.stats["pop"][-2:-1] or [info["pop"]])
        survival_frac = info["pop"]/(max(pop_before,1))

        if len(op.buf["obs"])>=CFG.rollout_len:
            mo=op.update(global_score=info["rew_o"],survival_frac=survival_frac)
        if len(wp.buf["obs"])>=CFG.rollout_len:
            mw=wp.update(global_score=info["rew_w"],survival_frac=survival_frac)

        # Periodic updates
        if step%CFG.pbt_interval==0:
            op.pbt_step(); wp.pbt_step()
        if step%CFG.neat_mutate_freq==0:
            op.neat_step(); wp.neat_step()
        if step%CFG.wm_train_freq==0:
            op.train_world_model(); wp.train_world_model()
        if step%CFG.selfplay_swap_interval==0:
            op.archive_policy(); wp.archive_policy()
        if step%CFG.curriculum_check_interval==0:
            diff=op.auto_curriculum()

        if step%50==0:
            el=time.time()-t0
            print(f"Step {step:5d} | "
                  f"Org {info['pop']:4d}({info['male']:3d}{info['female']:3d}"
                  f"V{info['in_v']:3d}E{info['eld']:2d}) | "
                  f"War {info['war']:3d} | Enm {info['enm']:2d}(G{Enemy.generation}) | "
                  f"V{info['n_v']} Sp{info['n_species']} | Fd {info['food']:4d} | "
                  f"B{info['births']:2d} K{info['killed']:2d} Br{info['breach']:2d} "
                  f"EK{info['ekills']:2d} Tr{info['trade']:2d} | "
                  f"Curr{info['curriculum']:.2f} | "
                  f"OPL {mo.get('pl',0):+.3f} | {el:.0f}s")

        if step%CFG.render_every==0: rend.render(world,step)

        if not world.organisms:
            respawns+=1
            print(f"\n  Extinction at step {step}. Respawn #{respawns}")
            if respawns<=10:
                for v in world.villages:
                    for k in range(20):
                        ang=random.uniform(0,2*math.pi); r=random.uniform(0,CFG.village_radius*0.7)
                        sex=Sex.MALE if k%2==0 else Sex.FEMALE
                        org=Organism(float(np.clip(v.x+r*math.cos(ang),0,CFG.world_w)),
                                     float(np.clip(v.y+r*math.sin(ang),0,CFG.world_h)),
                                     sex=sex,energy=CFG.init_energy*0.75,clan_id=v.clan_id,
                                     culture=v.culture.copy())
                        org.home_village=v.id; world.species_tracker.assign(org.genome)
                        world.organisms.append(org)
            else:
                print("   Too many extinctions."); break

        if not world.enemies and step<CFG.max_steps-300:
            world.enemies.append(Enemy(*world._rp()))

    rend.render(world,world.step_count)
    print("\n"+""*78)
    ao=world.organisms; aw=world.warriors
    males=[o for o in ao if o.sex==Sex.MALE]; females=[o for o in ao if o.sex==Sex.FEMALE]
    print(f"  Final: Orgs {len(ao)}({len(males)}{len(females)}) | War {len(aw)} | Enm {len(world.enemies)}")
    print(f"  Villages {len(world.villages)} | Species {len(world.species_tracker.species)} | Enemy Gen {Enemy.generation}")
    if ao:
        gens=[o.generation for o in ao]
        print(f"  Gen avg {np.mean(gens):.2f} max {max(gens)}")
        print(f"   speed {np.mean([o.genome.speed for o in males]):.3f}   speed {np.mean([o.genome.speed for o in females]):.3f}")
        profs={p.name:sum(1 for o in ao if o.profession==p) for p in Profession}
        print(f"  Professions: {profs}")
        govs={g.name:sum(1 for v in world.villages if v.governance==g) for g in Governance}
        print(f"  Governance: {govs}")
    print(f"  Total births: {sum(world.stats['births_org'])}")
    print(f"  Total enemy kills: {sum(world.stats['enemy_kills'])}")
    print(f"  Total trades: {sum(world.stats['trade_trips'])}")
    print(f"  PBT best: C{op.active_pbt} score {op.pbt_configs[op.active_pbt].score:.2f}")
    print(f"  NEAT best score: {max(op.neat.scores):.3f}")
    print(f"  Archive snapshots: {len(op.archive)}")
    print(""*78+"\n")
    fp=os.path.join(CFG.plot_dir,"final_summary_v6.png")
    rend.fig.savefig(fp,dpi=120,bbox_inches="tight",facecolor=rend.fig.get_facecolor())
    print(f"[Done]  {fp}")
    return world,op,wp,rend



