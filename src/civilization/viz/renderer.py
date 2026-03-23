from __future__ import annotations

import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..core.config import CFG, DisasterType, Governance, Profession, TerrainType, Sex
from ..core.entities import Enemy
from ..env.world import World
op_ref = []

class Renderer:
    def __init__(self):
        self.fig=plt.figure(figsize=(32,20),facecolor="#0a0c10")
        gs_layout=self.fig.add_gridspec(3,5,wspace=0.42,hspace=0.55)
        self.axes=[self.fig.add_subplot(gs_layout[r,c]) for r in range(3) for c in range(5)]
        os.makedirs(CFG.plot_dir,exist_ok=True) if CFG.save_plot else None
        self._frame=0

    def _sty(self,ax,title="",xl="",yl=""):
        ax.set_facecolor("#13161e"); ax.tick_params(colors="#7a8394",labelsize=6.5)
        ax.spines[:].set_color("#252a35")
        ax.set_title(title,color="#e8edf5",fontsize=7.5,pad=4,fontweight="bold")
        ax.set_xlabel(xl,color="#7a8394",fontsize=6.5); ax.set_ylabel(yl,color="#7a8394",fontsize=6.5)

    def render(self, world: World, step: int):
        for ax in self.axes: ax.cla()
        ao=world.organisms; aw=world.warriors
        ae=[e for e in world.enemies if e.alive]
        st=world.stats; xs=list(range(len(st["pop"])))
        sm=lambda s,n:(np.convolve(s,np.ones(n)/n,mode="same").tolist() if len(s)>=n else s)
        males=[o for o in ao if o.sex==Sex.MALE]; females=[o for o in ao if o.sex==Sex.FEMALE]
        elders=[o for o in ao if o.is_elder]; sick=[o for o in ao if o.sick]
        chiefs=[w for w in aw if w.is_war_chief]; preg=[o for o in ao if o.pregnant]
        traders=[o for o in ao if o.carrying_food]
        explorers=[o for o in ao if o.profession==Profession.EXPLORER]

        #  [0] World Map 
        ax=self.axes[0]; self._sty(ax,f"Civilization World [step {step}]","X","Y")
        ax.set_xlim(0,CFG.world_w); ax.set_ylim(0,CFG.world_h)

        # Terrain background
        tm=world.terrain
        for ty in range(tm.H):
            for tx in range(tm.W):
                t=tm.grid[ty,tx]
                col=TERRAIN_COLORS[t]
                rx=tx/tm.W*CFG.world_w; ry=ty/tm.H*CFG.world_h
                rw=CFG.world_w/tm.W; rh=CFG.world_h/tm.H
                ax.add_patch(plt.Rectangle((rx,ry),rw,rh,color=col,alpha=0.5,zorder=0))

        # Disaster zones
        for d in world.disaster_mgr.active:
            dcols={DisasterType.DROUGHT:"#8B4513",DisasterType.FLOOD:"#1a3aff",
                   DisasterType.FIRE:"#ff4500",DisasterType.PLAGUE:"#7f007f"}
            dc=dcols.get(d["type"],"#ff0000")
            ax.add_patch(plt.Circle((d["x"],d["y"]),d["radius"],color=dc,alpha=0.22,zorder=1))

        # Villages
        for v in world.villages:
            col="#30d158" if v.is_defended else "#ff9f0a"
            ax.add_patch(plt.Circle((v.x,v.y),CFG.territory_radius,color=col,alpha=0.04,zorder=1))
            ax.add_patch(plt.Circle((v.x,v.y),CFG.village_radius,color=col,alpha=0.15,zorder=2))
            ax.add_patch(plt.Circle((v.x,v.y),CFG.village_radius,fill=False,edgecolor=col,lw=1.8,alpha=0.8,zorder=3))
            txt=f"V{v.id}\n{v.governance.name[:4]}"+(f"\n" if not v.is_defended else "")+(f"\n" if v.plague_active else "")
            ax.text(v.x,v.y,txt,color=col,fontsize=4.5,ha="center",va="center",zorder=4,fontweight="bold")

        # Bad zones
        for z in world.bad_zones:
            ax.add_patch(plt.Circle(z,CFG.bad_zone_radius,color="#ff3b30",alpha=0.15,zorder=2))
            ax.add_patch(plt.Circle(z,CFG.bad_zone_radius,fill=False,edgecolor="#ff3b30",lw=1,alpha=0.5,zorder=3))

        # Food
        if world.food:
            fx,fy=zip(*world.food); ax.scatter(fx,fy,s=3,c="#30d158",alpha=0.35,zorder=4,marker=".",linewidths=0)

        # Organisms by sex + profession marker
        PROF_MARKERS={Profession.FARMER:"o",Profession.TRADER:"s",
                      Profession.EXPLORER:"^",Profession.HEALER:"+",Profession.WARRIOR:"D"}
        if males:
            me=np.array([o.energy/CFG.max_energy for o in males]); ms=4+12*me
            ax.scatter([o.x for o in males],[o.y for o in males],s=ms,c="#0a84ff",alpha=0.78,zorder=5,linewidths=0.1)
        if females:
            fe=np.array([o.energy/CFG.max_energy for o in females]); fs=5+13*fe
            ax.scatter([o.x for o in females],[o.y for o in females],s=fs,c="#ff375f",alpha=0.78,zorder=5,linewidths=0.1)
        if preg: ax.scatter([o.x for o in preg],[o.y for o in preg],s=11,c="#ffd60a",zorder=6,marker="*",linewidths=0)
        if elders: ax.scatter([o.x for o in elders],[o.y for o in elders],s=14,c="#bf5af2",zorder=6,marker="D",linewidths=0)
        if sick: ax.scatter([o.x for o in sick],[o.y for o in sick],s=9,c="#ff9f0a",zorder=6,marker="x",linewidths=0.7)
        if traders: ax.scatter([o.x for o in traders],[o.y for o in traders],s=8,c="#ffd60a",zorder=6,marker="s",linewidths=0)
        if explorers: ax.scatter([o.x for o in explorers],[o.y for o in explorers],s=8,c="#64d2ff",zorder=6,marker="^",linewidths=0)

        if aw: ax.scatter([w.x for w in aw],[w.y for w in aw],s=40,c="#30d158",marker="D",zorder=7,edgecolors="#fff",lw=0.4,alpha=0.9)
        if chiefs: ax.scatter([w.x for w in chiefs],[w.y for w in chiefs],s=65,c="#ffd60a",marker="*",zorder=8)
        for e in ae:
            sz=min(60+40*(Enemy.generation-1),200)
            ax.scatter(e.x,e.y,s=sz,c="#ff453a",marker="^",zorder=9,edgecolors="#fff",lw=0.4)

        leg=[mpatches.Patch(color="#0a84ff",label=f"({len(males)})"),
             mpatches.Patch(color="#ff375f",label=f"({len(females)})"),
             mpatches.Patch(color="#ffd60a",label=f"Preg({len(preg)})"),
             mpatches.Patch(color="#bf5af2",label=f"Elder({len(elders)})"),
             mpatches.Patch(color="#ff9f0a",label=f"Sick({len(sick)})"),
             mpatches.Patch(color="#ffd60a",label=f"Trade({len(traders)})"),
             mpatches.Patch(color="#30d158",label=f"War({len(aw)})"),
             mpatches.Patch(color="#ff453a",label=f"Enm({len(ae)})")]
        ax.legend(handles=leg,loc="upper right",fontsize=4.2,
                  facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35",ncol=2)

        #  [1] Population 
        ax=self.axes[1]; self._sty(ax,"Population Dynamics","Step","Count")
        ax.plot(xs,st["pop"],color="#ffffff",lw=1.3,label="Total")
        ax.plot(xs,st["pop_m"],color="#0a84ff",lw=1.0,label="")
        ax.plot(xs,st["pop_f"],color="#ff375f",lw=1.0,label="")
        ax.plot(xs,st["warriors"],color="#30d158",lw=1.0,label="War")
        ax.plot(xs,st["enemies"],color="#ff453a",lw=0.9,alpha=0.7,label="Enm")
        ax.legend(fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35",ncol=2)

        #  [2] Reproduction & Health 
        ax=self.axes[2]; self._sty(ax,"Health & Reproduction","Step","Count")
        ax.plot(xs,sm(st["pregnancies"],10),color="#ffd60a",lw=1.2,label="Pregnant")
        ax.plot(xs,sm(st["births_org"],10), color="#30d158",lw=1.2,label="Births")
        ax.plot(xs,sm(st["sick"],10),       color="#ff9f0a",lw=1.0,label="Sick")
        ax.plot(xs,sm(st["elders"],5),      color="#bf5af2",lw=0.9,label="Elders")
        ax.legend(fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [3] Villages & Trade 
        ax=self.axes[3]; self._sty(ax,"Villages & Trade","Step","Count")
        ax.plot(xs,st["n_villages"],color="#30d158",lw=1.4,label="Villages")
        ax.plot(xs,sm(st["in_village"],5),color="#64d2ff",lw=1.0,label="In Village")
        ax2=ax.twinx(); ax2.tick_params(colors="#7a8394",labelsize=6.5); ax2.spines[:].set_color("#252a35")
        ax2.plot(xs,sm(st["trade_trips"],10),color="#ff9f0a",lw=1.0,linestyle="--",label="Trades")
        ax.set_ylabel("Villages",color="#30d158",fontsize=6.5); ax2.set_ylabel("Trades",color="#ff9f0a",fontsize=6.5)
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h1+h2,l1+l2,fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [4] Species & Disasters 
        ax=self.axes[4]; self._sty(ax,"Species & Disasters","Step","Count")
        ax.plot(xs,st["n_species"],color="#bf5af2",lw=1.4,label="Species")
        ax2=ax.twinx(); ax2.tick_params(colors="#7a8394",labelsize=6.5); ax2.spines[:].set_color("#252a35")
        ax2.plot(xs,sm(st["n_disasters"],5),color="#ff453a",lw=1.0,linestyle="--",label="Disasters")
        ax.set_ylabel("Species",color="#bf5af2",fontsize=6.5); ax2.set_ylabel("Disasters",color="#ff453a",fontsize=6.5)
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h1+h2,l1+l2,fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [5] Deaths 
        ax=self.axes[5]; self._sty(ax,"Deaths & Combat","Step","Count")
        ax.plot(xs,sm(st["deaths_starve"],10),color="#ff9f0a",lw=1.0,label="Starve")
        ax.plot(xs,sm(st["deaths_killed"],10),color="#ff453a",lw=1.0,label="Killed")
        ax.plot(xs,sm(st["deaths_breach"],10),color="#ff375f",lw=0.9,linestyle="--",label="Breach")
        ax.plot(xs,sm(st["deaths_war"],10),   color="#0a84ff",lw=0.9,label="War")
        ax.plot(xs,sm(st["enemy_kills"],10),  color="#30d158",lw=1.1,label="Enm")
        ax.legend(fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [6] Enemy Evolution 
        ax=self.axes[6]; self._sty(ax,"Enemy Evolution","Step","")
        ax2=ax.twinx(); ax2.tick_params(colors="#7a8394",labelsize=6.5); ax2.spines[:].set_color("#252a35")
        ax.plot(xs,st["enemy_speed"],color="#ff453a",lw=1.3,label="Speed")
        ax2.plot(xs,st["enemy_kr"],color="#ff9f0a",lw=1.1,linestyle="--",label="Kill R")
        ax.set_ylabel("Speed",color="#ff453a",fontsize=6.5); ax2.set_ylabel("Kill R",color="#ff9f0a",fontsize=6.5)
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h1+h2,l1+l2,fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [7] Gene: Speed dimorphism 
        ax=self.axes[7]; self._sty(ax,"Speed Gene  vs ","Step","Speed")
        ax.plot(xs,st["avg_spd_m"],color="#0a84ff",lw=1.2,label=" Speed")
        ax.plot(xs,st["avg_spd_f"],color="#ff375f",lw=1.1,label=" Speed")
        ax.legend(fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [8] Cooperation & Immunity 
        ax=self.axes[8]; self._sty(ax,"Social Genes","Step","Value")
        ax.plot(xs,st["avg_cooperation"],color="#30d158",lw=1.2,label="Cooperation")
        ax.plot(xs,st["avg_immune"],     color="#64d2ff",lw=1.1,label="Immunity")
        ax.plot(xs,st["avg_vision"],     color="#ffd60a",lw=1.0,label="Vision")
        ax.legend(fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [9] Generation 
        ax=self.axes[9]; self._sty(ax,"Avg Generation","Step","Gen")
        ax.plot(xs,st["avg_gen"],color="#ffd60a",lw=1.4)
        if xs: ax.fill_between(xs,st["avg_gen"],alpha=0.12,color="#ffd60a")

        #  [10] Reward 
        ax=self.axes[10]; self._sty(ax,"MAPPO Reward / Step","Step","Reward")
        ax.plot(xs,sm(st["reward_org"],25),color="#ff9f0a",lw=1.2,label="Org")
        ax.plot(xs,sm(st["reward_war"],25),color="#30d158",lw=1.1,label="Warrior")
        ax.axhline(0,color="#252a35",lw=0.8,linestyle="--")
        ax.legend(fontsize=5,facecolor="#1a1d25",labelcolor="#e8edf5",edgecolor="#252a35")

        #  [11] World Model Loss 
        ax=self.axes[11]; self._sty(ax,"World Model (Dreamer) Loss","Step","Loss")
        ax.plot(xs,sm(st["wm_loss"],20),color="#bf5af2",lw=1.2)
        if xs: ax.fill_between(xs,sm(st["wm_loss"],20),alpha=0.10,color="#bf5af2")

        #  [12] Auto-Curriculum 
        ax=self.axes[12]; self._sty(ax,"Auto-Curriculum Difficulty","Step","Difficulty")
        ax.plot(xs,st["curriculum_diff"],color="#ff9f0a",lw=1.3)
        ax.axhline(1.0,color="#252a35",lw=0.8,linestyle="--")
        ax2=ax.twinx(); ax2.tick_params(colors="#7a8394",labelsize=6.5); ax2.spines[:].set_color("#252a35")
        ax2.plot(xs,st["season"],color="#64d2ff",lw=0.9,linestyle=":",label="Season")
        ax2.set_ylabel("Season",color="#64d2ff",fontsize=6.5)

        #  [13] Curiosity Heatmap 
        ax=self.axes[13]; self._sty(ax,"Curiosity Visit Map (Exploration)","X","Y")
        if op_ref:
            vm=op_ref[0].visit_grid
            ax.imshow(vm,aspect="auto",cmap="inferno",origin="lower",
                      extent=[0,CFG.world_w,0,CFG.world_h],alpha=0.92)
        ax.set_xlim(0,CFG.world_w); ax.set_ylim(0,CFG.world_h)
        # Overlay village positions
        for v in world.villages:
            ax.plot(v.x,v.y,"w.",markersize=4,zorder=5)

        #  [14] Civilization Summary 
        ax=self.axes[14]; self._sty(ax,"Civilization Status","","")
        ax.axis("off")
        op0=op_ref[0] if op_ref else None
        tc=sum(world.stats["trade_trips"]) if world.stats["trade_trips"] else 0
        wc=sum(w.kills for w in aw)
        govs={g.name:sum(1 for v in world.villages if v.governance==g) for g in Governance}
        spec_counts={sid:info["count"] for sid,info in world.species_tracker.species.items()}
        top_sids=sorted(spec_counts,key=spec_counts.get,reverse=True)[:3]
        lines=[
            f"Step: {step}",
            f"Births: {sum(st['births_org'])} | Kills: {sum(st['enemy_kills'])}",
            f"Trades: {tc} | Breach: {sum(st['deaths_breach'])}",
            f"Species: {len(world.species_tracker.species)}",
            f"  Top: {', '.join(f'S{s}({spec_counts[s]})' for s in top_sids)}",
            f"Villages: {len(world.villages)} | Enemy Gen: {Enemy.generation}",
            f"Governance: D{govs.get('DEMOCRACY',0)} A{govs.get('AUTOCRACY',0)} T{govs.get('THEOCRACY',0)}",
            f"Disasters active: {len(world.disaster_mgr.active)}",
            f"Curriculum: {world.stats['curriculum_diff'][-1]:.2f}" if world.stats['curriculum_diff'] else "",
            f"PBT active: C{op0.active_pbt if op0 else 0}",
            f"WM loss: {world.stats['wm_loss'][-1]:.4f}" if world.stats['wm_loss'] else "",
            f"Archive: {len(op0.archive) if op0 else 0} snapshots",
            f"NEAT best score: {max(op0.neat.scores):.3f}" if op0 else "",
            f"HRL goals: {len(set(op0.hrl.agent_goals.values())) if op0 else 0} active",
        ]
        for i,l in enumerate(lines):
            ax.text(0.04,0.97-i*0.067,l,transform=ax.transAxes,
                    color="#e8edf5",fontsize=7.2,va="top",family="monospace")

        self.fig.suptitle(
            f"Organism Living System v6    MAPPO+GRU+PBT+HRL+WorldModel+NEAT+Comm    Step {step}  |  "
            f"Orgs {len(ao)} ({len(males)} {len(females)})  |  "
            f"Warriors {len(aw)}  |  Enemies {len(ae)} (Gen {Enemy.generation})  |  "
            f"Villages {len(world.villages)}  |  Species {len(world.species_tracker.species)}  |  "
            f"Food {len(world.food)}  |  Season {world.season_phase:.2f}",
            color="#e8edf5",fontsize=9,fontweight="bold",y=1.002)
        plt.tight_layout()
        if CFG.save_plot:
            self.fig.savefig(os.path.join(CFG.plot_dir,f"frame_{self._frame:05d}.png"),
                             dpi=95,bbox_inches="tight",facecolor=self.fig.get_facecolor())
            self._frame+=1
        try: plt.pause(0.001)
        except: pass

  
#  MAIN
  


