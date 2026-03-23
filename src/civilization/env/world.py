from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..core.config import CFG, Profession, Sex, TERRAIN, Governance, TerrainType
from ..core.entities import DisasterManager, Enemy, Organism, SpeciesTracker, Village, Warrior
from ..rl.policy import CivPolicy
def build_global_state(world) -> np.ndarray:
    W,H=CFG.world_w,CFG.world_h
    ao=world.organisms; aw=world.warriors; ae=world.enemies
    gs=np.zeros(CFG.global_state_dim,dtype=np.float32)
    gs[0]=min(len(ao)/CFG.max_organisms,1.);    gs[1]=min(len(aw)/CFG.max_warriors,1.)
    gs[2]=min(len(ae)/CFG.max_enemies,1.);      gs[3]=min(len(world.food)/CFG.max_food,1.)
    gs[4]=min(len(world.villages)/CFG.max_villages,1.)
    gs[5]=Enemy.current_speed/CFG.enemy_max_speed_cap
    gs[6]=Enemy.current_kr/CFG.enemy_kill_radius_cap
    gs[7]=math.sin(world.season_phase*2*math.pi); gs[8]=math.cos(world.season_phase*2*math.pi)
    if ao:
        gs[9]=np.mean([o.x for o in ao])/W;     gs[10]=np.mean([o.y for o in ao])/H
        gs[11]=np.mean([o.energy for o in ao])/CFG.max_energy
        gs[12]=np.mean([o.genome.speed for o in ao])/4.5
        gs[13]=float(sum(1 for o in ao if o.sex==Sex.MALE))/max(len(ao),1)
        gs[14]=float(sum(1 for o in ao if o.pregnant))/max(len(ao),1)
        gs[15]=float(sum(1 for o in ao if o.is_elder))/max(len(ao),1)
        gs[16]=float(sum(1 for o in ao if o.sick))/max(len(ao),1)
        # Profession distribution
        for pi, prof in enumerate(Profession):
            gs[17+pi]=sum(1 for o in ao if o.profession==prof)/max(len(ao),1)
    if aw:
        gs[22]=np.mean([w.energy for w in aw])/CFG.warrior_energy_max
        gs[23]=float(sum(1 for w in aw if w.is_war_chief))/max(len(aw),1)
    gs[24]=min(len(world.species_tracker.species)/10.,1.)
    gs[25]=float(len(world.disaster_mgr.active))/5.
    gs[26]=world.op_policy.curriculum_difficulty if world.op_policy else 1.0
    if ae:
        gs[27]=np.mean([e.x for e in ae])/W; gs[28]=np.mean([e.y for e in ae])/H
    gs[29]=min(sum(v.resources[2] for v in world.villages)/100.,1.)   # total metal
    gs[30]=min(sum(v.resources[3] for v in world.villages)/50.,1.)    # total medicine
    gs[31]=float(world.step_count)/CFG.max_steps
    return gs

  
#  WORLD
  
class World:
    def __init__(self):
        Enemy.reset()
        self.terrain   = TERRAIN
        self.villages: List[Village]  = []
        self.organisms: List[Organism]= []
        self.warriors:  List[Warrior] = []
        self.enemies:   List[Enemy]   = []
        self.food:      List[Tuple]   = []
        self.bad_zones: List[Tuple]   = []
        self.step_count=0; self._last_village_step=-CFG.clan_form_cooldown
        self.season_phase=0.0
        self.species_tracker = SpeciesTracker()
        self.disaster_mgr    = DisasterManager()
        self.op_policy = None   # set by run()
        self.stats = {k:[] for k in [
            "pop","pop_m","pop_f","warriors","enemies","food",
            "in_village","n_villages","pregnancies","elders","sick",
            "births_org","births_war","deaths_starve","deaths_killed",
            "deaths_breach","deaths_war","enemy_kills","trade_trips",
            "avg_spd_m","avg_spd_f","avg_vision","avg_gen","avg_immune","avg_cooperation",
            "enemy_speed","enemy_kr","reward_org","reward_war","season",
            "n_species","n_disasters","wm_loss","curriculum_diff",
        ]}
        self._init()

    def _rp(self, margin=10):
        return (random.uniform(margin,CFG.world_w-margin),
                random.uniform(margin,CFG.world_h-margin))

    def _init(self):
        positions=[(CFG.world_w*0.28,CFG.world_h*0.5),(CFG.world_w*0.72,CFG.world_h*0.5)]
        for i in range(min(CFG.init_villages,2)):
            v=Village(*positions[i],clan_id=i+1); v.founded_step=0
            self.villages.append(v)
        att=0
        while len(self.bad_zones)<CFG.n_bad_zones and att<300:
            x,y=self._rp()
            if all(math.hypot(x-v.x,y-v.y)>CFG.village_radius+CFG.bad_zone_radius+10
                   for v in self.villages):
                self.bad_zones.append((x,y))
            att+=1
        for _ in range(CFG.init_food):
            x,y=self._rp()
            self.food.append((x,y))
        for vi,v in enumerate(self.villages):
            per=CFG.init_organisms//len(self.villages)
            for k in range(per):
                ang=random.uniform(0,2*math.pi); r=random.uniform(0,CFG.village_radius*0.85)
                x=float(np.clip(v.x+r*math.cos(ang),0,CFG.world_w))
                y=float(np.clip(v.y+r*math.sin(ang),0,CFG.world_h))
                sex=Sex.MALE if k%2==0 else Sex.FEMALE
                org=Organism(x,y,sex=sex,clan_id=v.clan_id,culture=v.culture.copy())
                org.home_village=v.id
                self.species_tracker.assign(org.genome)
                self.organisms.append(org)
            for k in range(CFG.init_warriors//len(self.villages)):
                ang=random.uniform(0,2*math.pi); r=random.uniform(0,CFG.village_radius)
                x=float(np.clip(v.x+r*math.cos(ang),0,CFG.world_w))
                y=float(np.clip(v.y+r*math.sin(ang),0,CFG.world_h))
                self.warriors.append(Warrior(x,y,home_village=v.id,clan_id=v.clan_id))
        for _ in range(CFG.init_enemies):
            for _ in range(100):
                edge=random.choice(["T","B","L","R"])
                if   edge=="T": x,y=random.uniform(0,CFG.world_w),random.uniform(0,8)
                elif edge=="B": x,y=random.uniform(0,CFG.world_w),random.uniform(CFG.world_h-8,CFG.world_h)
                elif edge=="L": x,y=random.uniform(0,8),random.uniform(0,CFG.world_h)
                else:           x,y=random.uniform(CFG.world_w-8,CFG.world_w),random.uniform(0,CFG.world_h)
                if all(math.hypot(x-v.x,y-v.y)>CFG.village_radius+12 for v in self.villages):
                    self.enemies.append(Enemy(x,y)); break

    def _update_season(self):
        self.season_phase=(self.step_count%CFG.season_period)/CFG.season_period
        return 1.0+CFG.season_amp*math.sin(self.season_phase*2*math.pi)

    def _update_defence(self, aw):
        for v in self.villages:
            v.is_defended=any(math.hypot(w.x-v.x,w.y-v.y)<=CFG.village_breach_radius for w in aw)

    def _auto_respawn_warriors(self, aw):
        for v in self.villages:
            local=[w for w in aw if w.home_village==v.id]
            if (len(local)<CFG.min_warriors_per_village
                    and self.step_count-v.last_warrior_spawn>=CFG.warrior_respawn_interval
                    and len(self.warriors)<CFG.max_warriors):
                ang=random.uniform(0,2*math.pi); r=random.uniform(0,CFG.village_radius*0.6)
                x=float(np.clip(v.x+r*math.cos(ang),0,CFG.world_w))
                y=float(np.clip(v.y+r*math.sin(ang),0,CFG.world_h))
                self.warriors.append(Warrior(x,y,energy=CFG.warrior_energy_init*0.8,
                                             home_village=v.id,clan_id=v.clan_id))
                v.last_warrior_spawn=self.step_count

    def _elect_war_chief(self, aw):
        for v in self.villages:
            vw=[w for w in aw if w.home_village==v.id]
            if not vw: continue
            best=max(vw,key=lambda w:w.kills)
            for w in vw: w.is_war_chief=False
            if best.kills>4: best.is_war_chief=True

    def _formation_combat(self, aw):
        """Assign warriors to squads, coordinate flanking."""
        squads: Dict[int, List[Warrior]] = defaultdict(list)
        for w in aw:
            if w.home_village is not None:
                squads[w.home_village].append(w)
        for vid, squad in squads.items():
            if len(squad) < 2: continue
            # Find centroid of squad
            cx = sum(w.x for w in squad)/len(squad)
            cy = sum(w.y for w in squad)/len(squad)
            # Assign flanking positions (surround nearest enemy)
            ae=[e for e in self.enemies if e.alive]
            if not ae: continue
            nearest_e = min(ae, key=lambda e:math.hypot(e.x-cx,e.y-cy))
            angles = [2*math.pi*i/len(squad) for i in range(len(squad))]
            for w, ang in zip(squad, angles):
                # Target position: surround enemy
                tx = nearest_e.x + math.cos(ang)*CFG.warrior_kill_radius*2
                ty = nearest_e.y + math.sin(ang)*CFG.warrior_kill_radius*2
                dx = tx-w.x; dy = ty-w.y; d=math.hypot(dx,dy)+1e-8
                if d>2:
                    spd=w.effective_speed()*0.8
                    w.vx += (dx/d)*spd*0.3; w.vy += (dy/d)*spd*0.3

    def _spread_disease(self, ao):
        for v in self.villages:
            local=[o for o in ao if o.alive and v.contains(o.x,o.y)]
            density=len(local)/max(1,(math.pi*CFG.village_radius**2))
            if density>0.10 and random.random()<0.0018:
                v.plague_active=True; v.plague_timer=220
            if v.plague_active:
                v.plague_timer-=1
                if v.plague_timer<=0: v.plague_active=False
                for o in local:
                    if not o.sick and random.random()<0.04*(1.-o.genome.immune_strength):
                        o.sick=True; o.sick_timer=CFG.sick_duration
            # Healers reduce sick count
            healers=[o for o in local if o.profession==Profession.HEALER and o.alive]
            for h in healers:
                nearby_sick=[o for o in local if o.sick and o.id!=h.id
                             and math.hypot(o.x-h.x,o.y-h.y)<8.]
                for s in nearby_sick[:2]:
                    s.sick_timer = max(0, s.sick_timer - 15)

    def _process_trade(self, ao) -> int:
        trips=0
        for o in ao:
            if not o.alive or not o.carrying_food: continue
            if o.trade_destination is None: o.carrying_food=False; continue
            dv=[v for v in self.villages if v.id==o.trade_destination]
            if not dv: o.carrying_food=False; o.trade_destination=None; continue
            v=dv[0]
            if v.dist(o.x,o.y)<=CFG.village_radius:
                v.resources[0]+=1.; o.carrying_food=False
                o.trade_destination=None
                bonus=CFG.trade_reward*o.profession_bonus().get("trade_reward",1.0)
                o.energy=min(o.energy+bonus,CFG.max_energy); trips+=1
        for o in ao:
            if not o.alive or o.carrying_food or o.pregnant: continue
            hv=[v for v in self.villages if v.id==o.home_village]
            if not hv: continue
            v=hv[0]
            if (random.random()<0.0025*o.genome.trade_drive*v.policy_trade
                    and len(self.villages)>1):
                others=[vv for vv in self.villages if vv.id!=v.id]
                if others:
                    o.carrying_food=True; o.trade_destination=random.choice(others).id
        return trips

    def _cultural_transmission(self, parent: Organism, child: Organism):
        """Child inherits blend of parent culture and village culture."""
        hv=[v for v in self.villages if v.id==parent.home_village]
        vil_culture = hv[0].culture if hv else np.random.rand(8).astype(np.float32)
        alpha = parent.genome.cultural_memory
        child.culture = (alpha*parent.culture + (1-alpha)*vil_culture
                         + np.random.randn(8).astype(np.float32)*0.02)
        child.culture = np.clip(child.culture, 0, 1)

    def _try_form_village(self, ao):
        if len(self.villages)>=CFG.max_villages: return
        if self.step_count-self._last_village_step<CFG.clan_form_cooldown: return
        for _ in range(80):
            cx=random.uniform(25,CFG.world_w-25); cy=random.uniform(25,CFG.world_h-25)
            if any(math.hypot(cx-v.x,cy-v.y)<CFG.village_radius*3.0 for v in self.villages): continue
            if any(math.hypot(cx-z[0],cy-z[1])<CFG.bad_zone_radius+CFG.village_radius for z in self.bad_zones): continue
            if self.terrain.get(cx,cy) in [TerrainType.MOUNTAIN, TerrainType.DESERT]: continue
            nearby=[o for o in ao if math.hypot(o.x-cx,o.y-cy)<=CFG.clan_form_radius]
            if len(nearby)>=CFG.clan_form_density:
                rcx=sum(o.x for o in nearby)/len(nearby); rcy=sum(o.y for o in nearby)/len(nearby)
                cc={}
                for o in nearby: c=o.clan_id or 0; cc[c]=cc.get(c,0)+1
                dom=max(cc,key=lambda k:cc[k])
                nv=Village(rcx,rcy,clan_id=dom); nv.founded_step=self.step_count
                # Inherit dominant culture
                dom_orgs=[o for o in nearby if o.clan_id==dom]
                if dom_orgs: nv.culture=np.mean([o.culture for o in dom_orgs],axis=0).astype(np.float32)
                self.villages.append(nv)
                for o in nearby:
                    if o.home_village is None or random.random()<0.5:
                        o.home_village=nv.id; o.clan_id=dom
                self._last_village_step=self.step_count
                print(f"  [Village+] ({rcx:.0f},{rcy:.0f}) clan={dom} governance={nv.governance.name} n={len(nearby)} total={len(self.villages)}")
                return

    def _handle_migration(self, ao):
        for v in self.villages:
            local=[o for o in ao if o.alive and v.contains(o.x,o.y)]
            if len(local)>CFG.overpop_threshold:
                migrants=random.sample(local,len(local)//4)
                candidates=[vv for vv in self.villages if vv.id!=v.id]
                if not candidates: continue
                target=random.choice(candidates)
                for m in migrants: m.home_village=target.id; m.clan_id=target.clan_id

    def _spawn_food(self, season_mult):
        ratio=len(self.food)/CFG.max_food
        for _ in range(3):  # try 3 spots per step
            if len(self.food)>=CFG.max_food: break
            x,y=self._rp()
            rate=CFG.food_spawn_rate*season_mult*max(0.,1.-ratio**1.3)*self.terrain.food_mult(x,y)
            if random.random()<rate/3.:
                self.food.append((x,y))

    def _org_threshold(self):
        fr=len(self.food)/max(CFG.max_food,1); return CFG.reproduce_threshold+(1.-fr)*22.

    def _org_rew(self, o, ate, in_bad, near_enemy, in_village, near_warrior, v_def,
                 curiosity, near_elder, near_healer):
        r  = 3.5 if ate else 0.
        r -= (2.5*(1.-o.genome.immune_strength)) if in_bad else 0.
        r -= 1.8 if (near_enemy and not in_village) else 0.
        r -= 0.6 if (near_enemy and in_village and not v_def) else 0.
        r += (1.0 if v_def else 0.4) if in_village else 0.
        r += 0.4 if near_warrior else 0.
        r += 0.6 if near_elder   else 0.
        r += 0.3 if near_healer  else 0.
        r += curiosity
        r += 0.09
        r -= 0.12*max(o.genome.speed-2.3,0)
        r -= 0.5 if o.sick else 0.
        # Profession-specific bonuses
        if o.profession==Profession.EXPLORER: r += 0.3
        if o.profession==Profession.TRADER and o.carrying_food: r += 0.2
        return float(r)

    def _war_rew(self, w, near_orgs, in_bad, v_undef, kills):
        r = 12.0*kills + 0.5*near_orgs
        r -= 2.5 if in_bad else 0.
        r += 2.5 if v_undef else 0.
        r += 0.12; return float(r)

    def _try_mate(self, ao, threshold):
        births=0
        males  =[o for o in ao if o.sex==Sex.MALE  and o.energy>=threshold and o.mate_timer==0]
        females=[o for o in ao if o.sex==Sex.FEMALE and o.energy>=threshold
                 and not o.pregnant and o.mate_timer==0]
        elders =[o for o in ao if o.is_elder]
        for f in females:
            if len(self.organisms)>=CFG.max_organisms: break
            near_elder=any(math.hypot(e.x-f.x,e.y-f.y)<=CFG.village_radius for e in elders)
            boost=CFG.elder_birth_bonus if near_elder else 0.
            if random.random()>f.genome.fertility+boost: continue
            cands=[(math.hypot(m.x-f.x,m.y-f.y),m) for m in males
                   if math.hypot(m.x-f.x,m.y-f.y)<=f.genome.vision*1.3]
            if not cands: continue
            _,father=min(cands,key=lambda t:t[0])
            f.pregnant=True; f.gestation=CFG.gestation_steps
            f.father_genome=father.genome
            f.energy-=CFG.reproduce_cost*0.58; father.energy-=CFG.reproduce_cost*0.42
            f.mate_timer=55; father.mate_timer=32
            if father in males: males.remove(father)
        for f in [o for o in ao if o.sex==Sex.FEMALE and o.pregnant]:
            f.gestation-=1
            if f.gestation<=0:
                f.pregnant=False
                if f.father_genome is None: continue
                csex=random.choice([Sex.MALE,Sex.FEMALE])
                cg=f.genome.crossover(f.father_genome,csex)
                child=Organism(
                    x=float(np.clip(f.x+random.uniform(-5,5),0,CFG.world_w)),
                    y=float(np.clip(f.y+random.uniform(-5,5),0,CFG.world_h)),
                    genome=cg,sex=csex,energy=CFG.reproduce_cost*0.42,
                    generation=f.generation+1,clan_id=f.clan_id)
                child.home_village=f.home_village
                self._cultural_transmission(f, child)
                self.species_tracker.assign(child.genome)
                f.father_genome=None
                self.organisms.append(child); births+=1
        return births

    def _reproduce_war(self, war):
        if war.energy<CFG.warrior_rep_threshold or len(self.warriors)>=CFG.max_warriors: return 0
        war.energy-=CFG.warrior_rep_cost
        self.warriors.append(Warrior(
            x=float(np.clip(war.x+random.uniform(-5,5),0,CFG.world_w)),
            y=float(np.clip(war.y+random.uniform(-5,5),0,CFG.world_h)),
            energy=CFG.warrior_rep_cost*0.5,
            home_village=war.home_village,clan_id=war.clan_id)); return 1

    def _enemy_dynamics(self, curriculum_diff):
        if self.step_count%CFG.enemy_growth_interval==0 and self.step_count>0:
            Enemy.grow(diff=curriculum_diff)
            print(f"  [Enemy] Spd={Enemy.current_speed:.2f} KR={Enemy.current_kr:.2f} Gen={Enemy.generation} diff={curriculum_diff:.2f}")
        if (self.step_count%CFG.enemy_spawn_interval==0 and self.step_count>0
                and len(self.enemies)<CFG.max_enemies):
            for _ in range(100):
                edge=random.choice(["T","B","L","R"])
                if   edge=="T": x,y=random.uniform(0,CFG.world_w),0.
                elif edge=="B": x,y=random.uniform(0,CFG.world_w),float(CFG.world_h)
                elif edge=="L": x,y=0.,random.uniform(0,CFG.world_h)
                else:           x,y=float(CFG.world_w),random.uniform(0,CFG.world_h)
                if all(math.hypot(x-v.x,y-v.y)>CFG.village_radius+6 for v in self.villages):
                    self.enemies.append(Enemy(x,y))
                    print(f"  [Enemy+] ({x:.0f},{y:.0f}) total={len(self.enemies)}"); break

    #  MAIN STEP 
    def step(self, op: CivPolicy, wp: CivPolicy) -> dict:
        self.step_count+=1
        self.op_policy = op
        season_mult=self._update_season()
        ao=[o for o in self.organisms if o.alive]
        aw=[w for w in self.warriors  if w.alive]
        ae=[e for e in self.enemies   if e.alive]
        births_o=0;births_w=0;d_starve=0;d_killed=0;d_breach=0;d_war=0;e_kills=0;trade=0
        rew_o=0.;rew_w=0.

        self._update_defence(aw)
        self._auto_respawn_warriors(aw)
        self._elect_war_chief(aw)
        for v in self.villages:
            v.update_governance(ao, aw)
            v.culture_drift()
        self._formation_combat(aw)

        gs = build_global_state(self)

        #  Nearby agents for communication 
        def get_nearby_ids(agent, all_agents, radius):
            return [a.id for a in all_agents
                    if a.id != agent.id and math.hypot(a.x-agent.x,a.y-agent.y)<=radius]

        #  Organism MAPPO act 
        elders  = [o for o in ao if o.is_elder]
        healers = [o for o in ao if o.profession==Profession.HEALER]
        obs_o   = [o.observe(self.food,ae,aw,self.villages,self.bad_zones,
                             elders,self.season_phase,self.terrain,
                             np.zeros(CFG.msg_dim,dtype=np.float32)) for o in ao]
        ids_o   = [o.id for o in ao]
        gs_list = [gs]*len(ao)
        nearby_o= [get_nearby_ids(o,ao,20.0) for o in ao]

        if obs_o:
            acts_o,logps_o,vals_o,hxs_o,goal_es,msg_es=op.batch_act(obs_o,ids_o,gs_list,nearby_o)
            for i,o in enumerate(ao):
                o._obs=obs_o[i]; o._act=int(acts_o[i])
                o._logp=float(logps_o[i]); o._val=float(vals_o[i])
        else:
            acts_o=logps_o=vals_o=hxs_o=goal_es=msg_es=[]

        #  Warrior MAPPO act 
        obs_w   = [w.observe(ae,ao,self.villages,self.bad_zones,self.season_phase,self.terrain) for w in aw]
        ids_w   = [w.id for w in aw]
        gs_wlist= [gs]*len(aw)
        nearby_w= [get_nearby_ids(w,aw,20.) for w in aw]

        if obs_w:
            acts_w,logps_w,vals_w,hxs_w,goal_ew,msg_ew=wp.batch_act(obs_w,ids_w,gs_wlist,nearby_w)
            for i,w in enumerate(aw):
                w._obs=obs_w[i]; w._act=int(acts_w[i])
                w._logp=float(logps_w[i]); w._val=float(vals_w[i])
        else:
            acts_w=logps_w=vals_w=hxs_w=goal_ew=msg_ew=[]

        #  Move organisms 
        for oi, o in enumerate(ao):
            fled=False
            for e in ae:
                d=math.hypot(e.x-o.x,e.y-o.y)
                if d<o.genome.vision*0.52:
                    dx=o.x-e.x;dy=o.y-e.y;dd=math.hypot(dx,dy)+1e-8
                    fx=dx/dd;fy=dy/dd;best=8;bdot=-99
                    for ai in range(8):
                        dot=_ACT_DIRS[ai][0]*fx+_ACT_DIRS[ai][1]*fy
                        if dot>bdot: bdot=dot;best=ai
                    o._act=best;fled=True;break
            # Trade movement
            if o.carrying_food and o.trade_destination and not fled:
                dv=[v for v in self.villages if v.id==o.trade_destination]
                if dv:
                    v=dv[0];dx=v.x-o.x;dy=v.y-o.y;dd=math.hypot(dx,dy)+1e-8
                    best=8;bdot=-99
                    for ai in range(8):
                        dot=_ACT_DIRS[ai][0]*(dx/dd)+_ACT_DIRS[ai][1]*(dy/dd)
                        if dot>bdot:bdot=dot;best=ai
                    o._act=best
            # Foraging
            in_v=any(v.contains(o.x,o.y) for v in self.villages)
            if in_v and o.energy>CFG.reproduce_threshold*0.72 and not fled:
                vis_food=[f for f in self.food if math.hypot(f[0]-o.x,f[1]-o.y)<=CFG.fog_radius]
                of=[(math.hypot(f[0]-o.x,f[1]-o.y),f) for f in vis_food
                    if not any(v.contains(f[0],f[1]) for v in self.villages)]
                if of:
                    _,tf=min(of,key=lambda t:t[0])
                    dx=tf[0]-o.x;dy=tf[1]-o.y;dd=math.hypot(dx,dy)+1e-8
                    best=8;bdot=-99
                    for ai in range(8):
                        dot=_ACT_DIRS[ai][0]*(dx/dd)+_ACT_DIRS[ai][1]*(dy/dd)
                        if dot>bdot:bdot=dot;best=ai
                    o._act=best
            # Return home
            hv=[v for v in self.villages if v.id==o.home_village]
            if hv and o.energy<CFG.init_energy*0.30 and not fled and not o.carrying_food:
                v=hv[0];dx=v.x-o.x;dy=v.y-o.y;dd=math.hypot(dx,dy)+1e-8
                best=8;bdot=-99
                for ai in range(8):
                    dot=_ACT_DIRS[ai][0]*(dx/dd)+_ACT_DIRS[ai][1]*(dy/dd)
                    if dot>bdot:bdot=dot;best=ai
                o._act=best
            o.apply_action(o._act, self.terrain)
            if not o.alive: d_starve+=1

        #  Move warriors 
        for w in aw:
            w.apply_action(w._act, self.terrain); w.override_charge(ae)
            if not w.alive: d_war+=1

        #  Move enemies 
        for e in ae: e.step(ao,aw,ae,self.villages,self.terrain)

        #  Eat food 
        food_set=list(self.food); taken=set()
        for o in ao:
            if not o.alive or o.carrying_food: continue
            for fi,f in enumerate(food_set):
                if fi in taken: continue
                er=(CFG.village_food_radius if any(v.contains(o.x,o.y) for v in self.villages)
                    else CFG.food_eat_radius)
                bonus=o.profession_bonus().get("food_mult",1.0)
                if math.hypot(f[0]-o.x,f[1]-o.y)<=er:
                    o.energy=min(o.energy+CFG.food_energy*bonus,CFG.max_energy)
                    taken.add(fi); break
        for w in aw:
            if not w.alive: continue
            for fi,f in enumerate(food_set):
                if fi in taken: continue
                if math.hypot(f[0]-w.x,f[1]-w.y)<=CFG.food_eat_radius:
                    w.energy=min(w.energy+CFG.warrior_food_energy,CFG.warrior_energy_max)
                    taken.add(fi); break
        self.food=[f for i,f in enumerate(food_set) if i not in taken]

        #  Warriors kill enemies 
        for w in aw:
            if w.alive: e_kills+=w.try_kill_enemies(ae)

        #  Bad zone + disaster damage 
        for o in ao:
            if o.alive and any(math.hypot(z[0]-o.x,z[1]-o.y)<=CFG.bad_zone_radius for z in self.bad_zones):
                dmg=CFG.bad_zone_damage*(1.-o.genome.immune_strength*0.5)
                o.energy-=dmg
                if o.energy<=0: o.alive=False; d_starve+=1

        #  Enemies kill organisms 
        for e in ae:
            if e.alive: k,br=e.try_kill(ao,self.villages); d_killed+=k; d_breach+=br

        #  Systems 
        self._spread_disease(ao)
        trade=self._process_trade(ao)
        self.disaster_mgr.tick(self, season_mult)

        #  World model store 
        for oi, o in enumerate(ao):
            if o._obs is not None and o.alive:
                op.world_model.store(o._obs, o._act, o.observe(
                    self.food,ae,aw,self.villages,self.bad_zones,
                    elders,self.season_phase,self.terrain,
                    np.zeros(CFG.msg_dim,dtype=np.float32)), 0.0)

        #  MAPPO rewards (organisms) 
        threshold=self._org_threshold()
        for oi, o in enumerate(ao):
            if o._obs is None: continue
            hid=op._get_hidden(o.id)
            in_bad=any(math.hypot(z[0]-o.x,z[1]-o.y)<=CFG.bad_zone_radius for z in self.bad_zones)
            n_enemy=any(math.hypot(e.x-o.x,e.y-o.y)<=o.genome.vision for e in ae if e.alive)
            in_vill=any(v.contains(o.x,o.y) for v in self.villages)
            n_war=any(math.hypot(w.x-o.x,w.y-o.y)<=CFG.warrior_protect_r for w in aw if w.alive)
            v_def=next((v.is_defended for v in self.villages if v.contains(o.x,o.y)),True)
            n_elder=any(math.hypot(e.x-o.x,e.y-o.y)<=CFG.village_radius for e in elders if e.id!=o.id)
            n_healer=any(math.hypot(h.x-o.x,h.y-o.y)<=CFG.village_radius for h in healers if h.id!=o.id)
            ate=o.energy>o._prev_energy
            cur=op.intrinsic_reward(o.x,o.y)
            rew=self._org_rew(o,ate,in_bad,n_enemy,in_vill,n_war,v_def,cur,n_elder,n_healer)
            o.fitness+=rew; rew_o+=rew
            ge_np = goal_es[oi] if oi < len(goal_es) else np.zeros(CFG.goal_embed_dim, dtype=np.float32)
            me_np = msg_es[oi]  if oi < len(msg_es)  else np.zeros(CFG.msg_dim, dtype=np.float32)
            if isinstance(me_np, torch.Tensor): me_np=me_np.cpu().numpy()
            op.store(o._obs,o._act,o._logp,o._val,rew,not o.alive,hid,gs,ge_np,me_np)

        #  MAPPO rewards (warriors) 
        for wi, w in enumerate(aw):
            if w._obs is None: continue
            hid=wp._get_hidden(w.id)
            in_bad=any(math.hypot(z[0]-w.x,z[1]-w.y)<=CFG.bad_zone_radius for z in self.bad_zones)
            n_orgs=sum(1 for o in ao if o.alive and math.hypot(o.x-w.x,o.y-w.y)<=CFG.warrior_protect_r)
            v_undef=any(not v.is_defended and v.id==w.home_village for v in self.villages)
            rew=self._war_rew(w,n_orgs,in_bad,v_undef,0)
            rew_w+=rew
            ge_np = goal_ew[wi] if wi < len(goal_ew) else np.zeros(CFG.goal_embed_dim,dtype=np.float32)
            me_np = msg_ew[wi]  if wi < len(msg_ew)  else np.zeros(CFG.msg_dim,dtype=np.float32)
            if isinstance(me_np, torch.Tensor): me_np=me_np.cpu().numpy()
            wp.store(w._obs,w._act,w._logp,w._val,rew,not w.alive,hid,gs,ge_np,me_np)

        #  Reproduction 
        births_o=self._try_mate(ao,threshold)
        for w in aw:
            if w.alive: births_w+=self._reproduce_war(w)

        #  Clear dead agent states 
        for o in ao:
            if not o.alive: op.clear_agent(o.id)
        for w in aw:
            if not w.alive: wp.clear_agent(w.id)

        #  Systems 
        self._spawn_food(season_mult)
        curriculum_diff = op.auto_curriculum()
        self._enemy_dynamics(curriculum_diff)
        self._try_form_village(self.organisms)
        self._handle_migration(self.organisms)
        self.species_tracker.prune(self.organisms)

        #  Prune dead 
        self.organisms=[o for o in self.organisms if o.alive]
        self.warriors =[w for w in self.warriors  if w.alive]
        self.enemies  =[e for e in self.enemies   if e.alive]

        #  Stats 
        ao2=self.organisms; aw2=self.warriors; ae2=self.enemies
        males=[o for o in ao2 if o.sex==Sex.MALE]; females=[o for o in ao2 if o.sex==Sex.FEMALE]
        in_v=sum(1 for o in ao2 if any(v.contains(o.x,o.y) for v in self.villages))
        preg=sum(1 for o in ao2 if o.pregnant); eld=sum(1 for o in ao2 if o.is_elder)
        sick=sum(1 for o in ao2 if o.sick)
        wm_l=float(np.mean(op.wm_loss_hist[-10:])) if op.wm_loss_hist else 0.
        self.stats["pop"].append(len(ao2)); self.stats["pop_m"].append(len(males))
        self.stats["pop_f"].append(len(females)); self.stats["warriors"].append(len(aw2))
        self.stats["enemies"].append(len(ae2)); self.stats["food"].append(len(self.food))
        self.stats["in_village"].append(in_v); self.stats["n_villages"].append(len(self.villages))
        self.stats["pregnancies"].append(preg); self.stats["elders"].append(eld); self.stats["sick"].append(sick)
        self.stats["births_org"].append(births_o); self.stats["births_war"].append(births_w)
        self.stats["deaths_starve"].append(d_starve); self.stats["deaths_killed"].append(d_killed)
        self.stats["deaths_breach"].append(d_breach); self.stats["deaths_war"].append(d_war)
        self.stats["enemy_kills"].append(e_kills); self.stats["trade_trips"].append(trade)
        self.stats["avg_spd_m"].append(np.mean([o.genome.speed for o in males]) if males else 0)
        self.stats["avg_spd_f"].append(np.mean([o.genome.speed for o in females]) if females else 0)
        self.stats["avg_vision"].append(np.mean([o.genome.vision for o in ao2]) if ao2 else 0)
        self.stats["avg_gen"].append(np.mean([o.generation for o in ao2]) if ao2 else 0)
        self.stats["avg_immune"].append(np.mean([o.genome.immune_strength for o in ao2]) if ao2 else 0)
        self.stats["avg_cooperation"].append(np.mean([o.genome.cooperation for o in ao2]) if ao2 else 0)
        self.stats["enemy_speed"].append(Enemy.current_speed); self.stats["enemy_kr"].append(Enemy.current_kr)
        self.stats["reward_org"].append(rew_o); self.stats["reward_war"].append(rew_w)
        self.stats["season"].append(season_mult); self.stats["n_species"].append(len(self.species_tracker.species))
        self.stats["n_disasters"].append(len(self.disaster_mgr.active))
        self.stats["wm_loss"].append(wm_l); self.stats["curriculum_diff"].append(curriculum_diff)
        return {"pop":len(ao2),"male":len(males),"female":len(females),
                "war":len(aw2),"enm":len(ae2),"food":len(self.food),
                "in_v":in_v,"n_v":len(self.villages),"preg":preg,"eld":eld,
                "births":births_o,"births_w":births_w,
                "killed":d_killed,"breach":d_breach,"d_war":d_war,
                "ekills":e_kills,"trade":trade,"rew_o":rew_o,"rew_w":rew_w,
                "n_species":len(self.species_tracker.species),"curriculum":curriculum_diff}

  
#  RENDERER  (15-panel civilization dashboard)
  


