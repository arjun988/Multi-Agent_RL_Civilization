from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CFG, Governance, Profession, Sex, TerrainMap, TerrainType, GoalType, DisasterType, _ACT_DIRS
from ..rl.policy import CivPolicy, Genome
_VID = 0
class Village:
    def __init__(self, x, y, clan_id=None):
        global _VID; _VID += 1
        self.id = _VID; self.x = float(x); self.y = float(y)
        self.clan_id = clan_id or _VID
        self.is_defended = True; self.founded_step = 0
        self.last_warrior_spawn = 0
        # Governance
        self.governance = random.choice(list(Governance))
        self.leader_id: Optional[int] = None
        self.policy_attack = random.random()    # 0=pacifist, 1=aggressive
        self.policy_trade  = random.random()
        self.policy_expand = random.random()
        # Multi-resource stockpile: food, wood, metal, medicine
        self.resources = np.array([10., 5., 2., 1.], dtype=np.float32)
        # Culture vector (transmitted to offspring)
        self.culture = np.random.rand(8).astype(np.float32)
        # Disease
        self.plague_active = False; self.plague_timer = 0
        # Disaster tracking
        self.disaster: Optional[DisasterType] = None
        self.disaster_timer = 0
        self.species_counts: Dict[int, int] = {}

    def contains(self, x, y): return math.hypot(x-self.x,y-self.y)<=CFG.village_radius
    def in_territory(self, x, y): return math.hypot(x-self.x,y-self.y)<=CFG.territory_radius
    def dist(self, x, y): return math.hypot(x-self.x,y-self.y)

    def update_governance(self, organisms, warriors):
        """Village governance affects policy."""
        local_orgs = [o for o in organisms if o.alive and self.contains(o.x,o.y)]
        local_wars = [w for w in warriors if w.alive and w.home_village==self.id]
        if not local_orgs: return
        if self.governance == Governance.DEMOCRACY:
            # Policy is average preference of residents
            self.policy_trade  = float(np.mean([o.genome.trade_drive for o in local_orgs]))
            self.policy_attack = float(np.mean([o.genome.aggression  for o in local_orgs]))
        elif self.governance == Governance.AUTOCRACY:
            # Leader sets policy
            if self.leader_id:
                leaders = [o for o in local_orgs if o.id == self.leader_id]
                if leaders:
                    ldr = leaders[0]
                    self.policy_trade  = ldr.genome.trade_drive
                    self.policy_attack = ldr.genome.aggression
        elif self.governance == Governance.THEOCRACY:
            # Elders dominate
            elders = [o for o in local_orgs if o.is_elder]
            if elders:
                self.policy_trade = float(np.mean([e.genome.trade_drive for e in elders]))
        # Elect leader: highest fitness in village
        if local_orgs:
            new_ldr = max(local_orgs, key=lambda o: o.fitness)
            self.leader_id = new_ldr.id

    def culture_drift(self):
        """Culture evolves slowly over time."""
        noise = np.random.randn(8).astype(np.float32) * 0.005
        self.culture = np.clip(self.culture + noise, 0, 1)

  
#  ORGANISM
  
_OID = 0
class Organism:
    def __init__(self, x, y, genome=None, energy=None, generation=1,
                 sex=None, clan_id=None, culture=None):
        global _OID; _OID += 1
        self.id=_OID; self.x=float(x); self.y=float(y)
        self.vx=0.; self.vy=0.
        self.sex=sex or random.choice([Sex.MALE,Sex.FEMALE])
        self.genome=genome or Genome.random(self.sex)
        self.energy=float(energy or CFG.init_energy)
        self.alive=True; self.age=0; self.generation=generation; self.fitness=0.
        self.home_village: Optional[int]=None; self.clan_id=clan_id
        self.profession=self._assign_profession()
        self.pregnant=False; self.gestation=0; self.father_genome=None
        self.mate_timer=0
        self.is_elder=False
        self.sick=False; self.sick_timer=0
        self.carrying_food=False
        self.trade_destination: Optional[int]=None
        # Cultural memory (inherited + learned)
        self.culture = culture if culture is not None else np.random.rand(8).astype(np.float32)
        # HRL goal state
        self.current_goal = GoalType.FORAGE
        self.goal_timer = 0
        # Fog of war: seen food positions (short-term episodic memory)
        self.known_food: deque = deque(maxlen=20)
        self._obs=None; self._act=8; self._logp=0.; self._val=0.
        self._prev_energy=self.energy

    def _assign_profession(self) -> Profession:
        """Profession from genes."""
        g = self.genome
        if g.aggression > 0.65:     return Profession.WARRIOR
        if g.trade_drive > 0.65:    return Profession.TRADER
        if g.curiosity_drive > 0.65:return Profession.EXPLORER
        if g.immune_strength > 0.65:return Profession.HEALER
        return Profession.FARMER

    def profession_bonus(self) -> Dict[str, float]:
        """Stat bonuses from profession."""
        if self.profession == Profession.FARMER:
            return {"food_mult": 1.3, "meta_mult": 0.85}
        if self.profession == Profession.TRADER:
            return {"trade_reward": 1.5}
        if self.profession == Profession.WARRIOR:
            return {"speed_mult": 1.15, "aggro_mult": 1.2}
        if self.profession == Profession.EXPLORER:
            return {"vision_mult": 1.25, "curiosity": 1.5}
        if self.profession == Profession.HEALER:
            return {"sick_resist": 1.5, "heal_nearby": 0.5}
        return {}

    # 56-dim observation
    def observe(self, food, enemies, warriors, villages, bad_zones,
                elders, season_phase, terrain: TerrainMap,
                agg_msg: np.ndarray) -> np.ndarray:
        W,H=CFG.world_w,CFG.world_h
        obs=np.zeros(CFG.org_obs_dim,dtype=np.float32)
        obs[0]=self.x/W; obs[1]=self.y/H
        obs[2]=self.energy/CFG.max_energy; obs[3]=min(self.age/700.,1.)
        obs[4]=float(self.sex==Sex.FEMALE)
        obs[5:17]=self.genome.to_vec()   # 12 genes

        def near(items,gx,gy):
            if not items: return 0.,0.,1.,0.
            d2=[(math.hypot(gx(i)-self.x,gy(i)-self.y),i) for i in items]
            d,it=min(d2,key=lambda t:t[0]); ix,iy=gx(it),gy(it)
            return (np.clip((ix-self.x)/W,-1,1),np.clip((iy-self.y)/H,-1,1),
                    min(d/self.genome.vision,1.),float(d<=self.genome.vision))

        # Only include food within fog radius
        vis_food=[f for f in food if math.hypot(f[0]-self.x,f[1]-self.y)<=CFG.fog_radius]
        obs[17],obs[18],obs[19],obs[20]=near(vis_food, lambda f:f[0],lambda f:f[1])
        obs[21],obs[22],obs[23],obs[24]=near(enemies,  lambda e:e.x, lambda e:e.y)
        obs[25],obs[26],obs[27],obs[28]=near(warriors, lambda w:w.x, lambda w:w.y)

        hv=[v for v in villages if v.id==self.home_village]
        if hv:
            v=hv[0]; d=v.dist(self.x,self.y)
            obs[29]=np.clip((v.x-self.x)/W,-1,1); obs[30]=np.clip((v.y-self.y)/H,-1,1)
            obs[31]=min(d/(W+H),1.); obs[32]=float(d<=CFG.village_radius)
            obs[33]=float(v.is_defended); obs[34]=float(v.plague_active)
            obs[35]=float(v.policy_attack); obs[36]=float(v.policy_trade)
            # Resource levels (normalised)
            res_norm=v.resources/np.array([50.,30.,15.,10.])
            obs[37:41]=np.clip(res_norm,0,1)

        if bad_zones:
            dsts=[math.hypot(z[0]-self.x,z[1]-self.y) for z in bad_zones]
            obs[41]=min(min(dsts)/W,1.); obs[42]=float(min(dsts)<=CFG.bad_zone_radius)

        obs[43]=np.clip(self.vx/4.,-1,1); obs[44]=np.clip(self.vy/4.,-1,1)
        obs[45]=min(len(vis_food)/14.,1.)
        obs[46]=np.clip((self.energy-self._prev_energy)/CFG.max_energy,-1,1)
        obs[47]=float(self.pregnant); obs[48]=float(self.sick)
        obs[49]=float(self.carrying_food)
        obs[50]=math.sin(season_phase*2*math.pi); obs[51]=math.cos(season_phase*2*math.pi)
        # Terrain info
        t=terrain.get(self.x,self.y)
        obs[52]=float(t==TerrainType.FOREST); obs[53]=float(t==TerrainType.DESERT)
        obs[54]=float(t==TerrainType.MOUNTAIN); obs[55]=float(t==TerrainType.RIVER)
        # Communication message embedded into obs
        # (msg included separately as goal_embed in policy)
        return obs

    def apply_action(self, action, terrain: TerrainMap):
        d=_ACT_DIRS[action]; spd=self.genome.speed
        if self.pregnant: spd*=0.72
        if self.sick:     spd*=0.78
        spd *= terrain.speed_mult(self.x,self.y)
        prof_bonus = self.profession_bonus()
        spd *= prof_bonus.get("speed_mult", 1.0)
        self.vx=self.vx*CFG.org_friction+d[0]*CFG.org_accel*spd
        self.vy=self.vy*CFG.org_friction+d[1]*CFG.org_accel*spd
        v=math.hypot(self.vx,self.vy)
        if v>spd: self.vx*=spd/v; self.vy*=spd/v
        self._prev_energy=self.energy
        self.x=float(np.clip(self.x+self.vx,0,CFG.world_w))
        self.y=float(np.clip(self.y+self.vy,0,CFG.world_h))
        if self.x<=0 or self.x>=CFG.world_w: self.vx*=-0.5
        if self.y<=0 or self.y>=CFG.world_h: self.vy*=-0.5
        meta_mult = prof_bonus.get("meta_mult", 1.0)
        cost=(CFG.base_metabolism*(1.+self.genome.metabolism)*meta_mult
              +CFG.move_cost_factor*(self.vx**2+self.vy**2))
        if self.pregnant: cost*=1.18
        if self.sick:
            cost*=1.12; self.energy-=CFG.sick_damage
            self.sick_timer-=1
            if self.sick_timer<=0: self.sick=False
        self.energy-=cost; self.age+=1
        if self.mate_timer>0: self.mate_timer-=1
        if self.age>=CFG.elder_age: self.is_elder=True
        if self.energy<=CFG.starve_energy: self.alive=False

  
#  WARRIOR
  
_WID = 0
class Warrior:
    def __init__(self, x, y, energy=None, home_village=None, clan_id=None):
        global _WID; _WID+=1
        self.id=_WID; self.x=float(x); self.y=float(y)
        self.vx=0.; self.vy=0.
        self.energy=float(energy or CFG.warrior_energy_init)
        self.alive=True; self.age=0; self.kills=0
        self.home_village=home_village; self.clan_id=clan_id
        self.is_war_chief=False
        self.squad_id: Optional[int]=None  # formation combat
        self._obs=None; self._act=8; self._logp=0.; self._val=0.

    def effective_speed(self):
        spd = CFG.warrior_max_speed*(1.+CFG.war_chief_bonus) if self.is_war_chief else CFG.warrior_max_speed
        return spd

    def effective_kill_radius(self):
        return (CFG.warrior_kill_radius*(1.+CFG.war_chief_bonus*0.5)
                if self.is_war_chief else CFG.warrior_kill_radius)

    def observe(self, enemies, organisms, villages, bad_zones,
                season_phase: float, terrain: TerrainMap) -> np.ndarray:
        W,H=CFG.world_w,CFG.world_h
        obs=np.zeros(CFG.warrior_obs_dim,dtype=np.float32)
        obs[0]=self.x/W; obs[1]=self.y/H
        obs[2]=self.energy/CFG.warrior_energy_max; obs[3]=min(self.age/600.,1.)

        def ns(items,gx,gy,k=0):
            if not items: return 0.,0.,1.,0.
            d2=[(math.hypot(gx(i)-self.x,gy(i)-self.y),i) for i in items]
            d2.sort(key=lambda t:t[0])
            if k>=len(d2): return 0.,0.,1.,0.
            d,it=d2[k]; ix,iy=gx(it),gy(it)
            return (np.clip((ix-self.x)/W,-1,1),np.clip((iy-self.y)/H,-1,1),
                    min(d/CFG.warrior_vision,1.),float(d<=CFG.warrior_vision))

        obs[4],obs[5],obs[6],obs[7]    =ns(enemies,  lambda e:e.x,lambda e:e.y,0)
        obs[8],obs[9],obs[10],obs[11]  =ns(organisms,lambda o:o.x,lambda o:o.y,0)
        obs[12],obs[13],obs[14],obs[15]=ns(enemies,  lambda e:e.x,lambda e:e.y,1)
        obs[16],obs[17],obs[18],obs[19]=ns(enemies,  lambda e:e.x,lambda e:e.y,2)  # 3rd enemy

        mv=[v for v in villages if v.id==self.home_village]
        if mv:
            v=mv[0]
            obs[20]=np.clip((v.x-self.x)/W,-1,1); obs[21]=np.clip((v.y-self.y)/H,-1,1)
            obs[22]=float(v.is_defended); obs[23]=float(v.plague_active)
            obs[24]=float(v.policy_attack); obs[25]=float(v.policy_trade)

        obs[26]=np.clip(self.vx/self.effective_speed(),-1,1)
        obs[27]=np.clip(self.vy/self.effective_speed(),-1,1)
        obs[28]=min(sum(1 for e in enemies
                        if math.hypot(e.x-self.x,e.y-self.y)<=CFG.warrior_vision)/8.,1.)
        obs[29]=min(sum(1 for o in organisms
                        if o.alive and math.hypot(o.x-self.x,o.y-self.y)<=CFG.warrior_protect_r)/14.,1.)
        obs[30]=min(self.kills/40.,1.); obs[31]=self.energy/CFG.warrior_energy_max
        obs[32]=float(self.is_war_chief)
        obs[33]=float(any(not v.is_defended and v.id==self.home_village for v in villages))
        obs[34]=math.sin(season_phase*2*math.pi); obs[35]=math.cos(season_phase*2*math.pi)
        t=terrain.get(self.x,self.y)
        obs[36]=float(t==TerrainType.MOUNTAIN); obs[37]=float(t==TerrainType.FOREST)
        return obs

    def apply_action(self, action, terrain: TerrainMap):
        spd=self.effective_speed()*terrain.speed_mult(self.x,self.y)
        d=_ACT_DIRS[action]
        self.vx=self.vx*CFG.warrior_friction+d[0]*CFG.warrior_accel*spd
        self.vy=self.vy*CFG.warrior_friction+d[1]*CFG.warrior_accel*spd
        v=math.hypot(self.vx,self.vy)
        if v>spd: self.vx*=spd/v; self.vy*=spd/v
        self.x=float(np.clip(self.x+self.vx,0,CFG.world_w))
        self.y=float(np.clip(self.y+self.vy,0,CFG.world_h))
        if self.x<=0 or self.x>=CFG.world_w: self.vx*=-0.5
        if self.y<=0 or self.y>=CFG.world_h: self.vy*=-0.5
        self.energy-=CFG.warrior_metabolism+CFG.warrior_move_cost*(self.vx**2+self.vy**2)
        self.age+=1
        if self.energy<=0: self.alive=False

    def override_charge(self, enemies):
        ae=[e for e in enemies if e.alive]
        if not ae: return
        d,tgt=min([(math.hypot(e.x-self.x,e.y-self.y),e) for e in ae],key=lambda t:t[0])
        if d<=CFG.warrior_vision:
            dx=tgt.x-self.x; dy=tgt.y-self.y; dd=math.hypot(dx,dy)+1e-8
            spd=self.effective_speed()
            self.vx=(dx/dd)*spd; self.vy=(dy/dd)*spd
            self.x=float(np.clip(self.x+self.vx,0,CFG.world_w))
            self.y=float(np.clip(self.y+self.vy,0,CFG.world_h))

    def try_kill_enemies(self, enemies):
        killed=0
        for e in enemies:
            if e.alive and math.hypot(e.x-self.x,e.y-self.y)<=self.effective_kill_radius():
                e.alive=False; self.kills+=1
                self.energy=min(self.energy+CFG.warrior_attack_bonus,CFG.warrior_energy_max)
                killed+=1
        return killed

  
#  ENEMY  (auto-curriculum adjusted)
  
_EID=0
class Enemy:
    current_speed=CFG.enemy_max_speed_init; current_kr=CFG.enemy_kill_radius_init
    current_vision=CFG.enemy_vision_init; generation=1

    @classmethod
    def grow(cls, diff=1.0):
        rate=CFG.enemy_growth_rate*diff
        cls.current_speed =min(cls.current_speed*(1+rate), CFG.enemy_max_speed_cap)
        cls.current_kr    =min(cls.current_kr*(1+rate),    CFG.enemy_kill_radius_cap)
        cls.current_vision=min(cls.current_vision*(1+rate*0.5), 26.)
        cls.generation+=1

    @classmethod
    def reset(cls):
        cls.current_speed=CFG.enemy_max_speed_init; cls.current_kr=CFG.enemy_kill_radius_init
        cls.current_vision=CFG.enemy_vision_init; cls.generation=1

    def __init__(self,x,y):
        global _EID; _EID+=1
        self.id=_EID; self.x=float(x); self.y=float(y)
        self.vx=0.; self.vy=0.; self.alive=True; self.age=0

    @property
    def speed(self): return Enemy.current_speed
    @property
    def kill_radius(self): return Enemy.current_kr
    @property
    def vision(self): return Enemy.current_vision

    def step(self, organisms, warriors, all_enemies, villages, terrain: TerrainMap):
        ao=[o for o in organisms if o.alive]; aw=[w for w in warriors if w.alive]
        fx,fy=0.,0.
        for w in aw:
            d=math.hypot(w.x-self.x,w.y-self.y)
            if d<15.: rx=(self.x-w.x)/(d+1e-8); ry=(self.y-w.y)/(d+1e-8); fx+=rx*(15.-d); fy+=ry*(15.-d)
        flee_len=math.hypot(fx,fy)
        cx,cy=0.,0.
        # Target weakest visible organism
        cands=[(math.hypot(o.x-self.x,o.y-self.y),o) for o in ao
               if math.hypot(o.x-self.x,o.y-self.y)<=self.vision]
        if cands:
            cands.sort(key=lambda t:t[1].energy); tgt=cands[0][1]
            dx=tgt.x-self.x; dy=tgt.y-self.y; d=math.hypot(dx,dy)+1e-8; cx=dx/d; cy=dy/d
        elif ao:
            tgt=random.choice(ao); dx=tgt.x-self.x; dy=tgt.y-self.y
            d=math.hypot(dx,dy)+1e-8; cx=dx/d*0.35; cy=dy/d*0.35
        # Pack cohesion
        px,py=0.,0.
        pack=[e for e in all_enemies if e is not self and e.alive
              and math.hypot(e.x-self.x,e.y-self.y)<CFG.enemy_pack_radius]
        if pack:
            pcx=sum(e.x for e in pack)/len(pack); pcy=sum(e.y for e in pack)/len(pack)
            dd=math.hypot(pcx-self.x,pcy-self.y)+1e-8
            px=(pcx-self.x)/dd*0.18; py=(pcy-self.y)/dd*0.18
        if flee_len>0:
            fx/=flee_len; fy/=flee_len; mx=0.65*fx+0.25*cx+0.1*px; my=0.65*fy+0.25*cy+0.1*py
        else: mx=0.85*cx+0.15*px; my=0.85*cy+0.15*py
        ml=math.hypot(mx,my)+1e-8
        spd=self.speed*terrain.speed_mult(self.x,self.y)
        ax=(mx/ml)*spd; ay=(my/ml)*spd
        self.vx=self.vx*CFG.enemy_friction+ax*CFG.enemy_accel
        self.vy=self.vy*CFG.enemy_friction+ay*CFG.enemy_accel
        v=math.hypot(self.vx,self.vy)
        if v>spd: self.vx*=spd/v; self.vy*=spd/v
        self.x=float(np.clip(self.x+self.vx,0,CFG.world_w))
        self.y=float(np.clip(self.y+self.vy,0,CFG.world_h))
        if self.x<=0 or self.x>=CFG.world_w: self.vx*=-0.6
        if self.y<=0 or self.y>=CFG.world_h: self.vy*=-0.6
        self.age+=1
        for v in villages:
            if v.contains(self.x,self.y) and v.is_defended:
                dx=self.x-v.x; dy=self.y-v.y; dd=math.hypot(dx,dy)+1e-8
                push=CFG.village_radius+0.5
                self.x=float(np.clip(v.x+(dx/dd)*push,0,CFG.world_w))
                self.y=float(np.clip(v.y+(dy/dd)*push,0,CFG.world_h))
                self.vx*=-0.3; self.vy*=-0.3

    def try_kill(self, organisms, villages):
        killed=0; breach=0
        for o in organisms:
            if not o.alive: continue
            d=math.hypot(o.x-self.x,o.y-self.y)
            if d>self.kill_radius: continue
            in_v=False; v_def=False
            for v in villages:
                if v.contains(o.x,o.y): in_v=True; v_def=v.is_defended; break
            if in_v and v_def: continue
            elif in_v and not v_def:
                if random.random()<CFG.village_breach_kill_prob: o.alive=False; killed+=1; breach+=1
            else: o.alive=False; killed+=1
        return killed,breach

  
#  SPECIATION TRACKER
  
class SpeciesTracker:
    def __init__(self):
        self.species: Dict[int, Dict] = {}  # species_id  {centroid, count}
        self._next_id = 1

    def assign(self, genome: Genome) -> int:
        """Assign or create species for genome."""
        if not self.species:
            sid = self._next_id; self._next_id += 1
            self.species[sid] = {"centroid": genome, "count": 1}
            genome.species_id = sid; return sid
        best_d = float("inf"); best_sid = -1
        for sid, info in self.species.items():
            d = genome.distance(info["centroid"])
            if d < best_d: best_d = d; best_sid = sid
        if best_d > CFG.speciation_threshold:
            sid = self._next_id; self._next_id += 1
            self.species[sid] = {"centroid": copy.copy(genome), "count": 1}
            print(f"  [Species] New species {sid} diverged! dist={best_d:.2f}")
        else:
            sid = best_sid; self.species[sid]["count"] += 1
        genome.species_id = sid; return sid

    def prune(self, organisms):
        """Update species counts."""
        counts: Dict[int,int] = defaultdict(int)
        for o in organisms: counts[o.genome.species_id] += 1
        self.species = {sid: {"centroid": info["centroid"], "count": counts.get(sid,0)}
                        for sid, info in self.species.items() if counts.get(sid,0) > 0}

  
#  NATURAL DISASTER MANAGER
  
class DisasterManager:
    def __init__(self):
        self.active: List[Dict] = []

    def tick(self, world, season_mult: float):
        # Spawn new disasters
        base_prob = CFG.disaster_prob
        if season_mult < 0.8: base_prob *= 2.0  # winter more dangerous
        if random.random() < base_prob:
            dtype = random.choice(list(DisasterType))
            cx = random.uniform(20, CFG.world_w-20)
            cy = random.uniform(20, CFG.world_h-20)
            r  = random.uniform(15, 35)
            self.active.append({
                "type": dtype, "x": cx, "y": cy, "radius": r,
                "timer": CFG.disaster_duration,
                "intensity": random.uniform(0.5, 1.5)
            })
            print(f"  [Disaster] {dtype.name} at ({cx:.0f},{cy:.0f}) r={r:.0f}")

        # Apply disasters
        for d in self.active:
            d["timer"] -= 1
            if d["type"] == DisasterType.DROUGHT:
                # Kill food in radius
                world.food = [f for f in world.food
                              if math.hypot(f[0]-d["x"],f[1]-d["y"]) > d["radius"]]
            elif d["type"] == DisasterType.FLOOD:
                # Damage orgs in radius
                for o in world.organisms:
                    if o.alive and math.hypot(o.x-d["x"],o.y-d["y"]) < d["radius"]:
                        o.energy -= 0.15 * d["intensity"]
                        if o.energy<=0: o.alive=False
            elif d["type"] == DisasterType.FIRE:
                # Kill food and damage anyone in radius
                world.food = [f for f in world.food
                              if math.hypot(f[0]-d["x"],f[1]-d["y"]) > d["radius"]*0.5]
                for o in world.organisms:
                    if o.alive and math.hypot(o.x-d["x"],o.y-d["y"]) < d["radius"]*0.7:
                        o.energy -= 0.08 * d["intensity"]
                        if o.energy<=0: o.alive=False
            elif d["type"] == DisasterType.PLAGUE:
                for o in world.organisms:
                    if o.alive and math.hypot(o.x-d["x"],o.y-d["y"]) < d["radius"]:
                        if not o.sick and random.random() < 0.03*(1-o.genome.immune_strength):
                            o.sick=True; o.sick_timer=CFG.sick_duration

        self.active = [d for d in self.active if d["timer"] > 0]

  
#  GLOBAL STATE  (centralised critic input)
  


