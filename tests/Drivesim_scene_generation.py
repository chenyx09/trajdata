from collections import namedtuple
from typing import Any, List, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene
from trajdata import AgentType
from trajdata.data_structures.agent import FixedExtent
from trajdata.caching.scene_cache import SceneCache
from trajdata.caching.env_cache import EnvCache
from trajdata.caching.df_cache import DataFrameCache
from trajdata.dataset_specific.scene_records import DrivesimSceneRecord
from pathlib import Path
import dill
from trajdata import MapAPI, VectorMap
import tbsim.utils.lane_utils as LaneUtils
from bokeh.plotting import figure, show, save
import bokeh

def generate_agentmeta(initial_state,agent_names,agent_extents,T,dt,hist_types):
    total_agent_data = list()
    total_agent_metadata = list()
    for x0,name,extent,htype in zip(initial_state,agent_names,agent_extents,hist_types):
        agent_meta = AgentMetadata(name=name,
                                   agent_type= AgentType.VEHICLE,
                                   first_timestep = 0,
                                   last_timestep = T-1,
                                   extent=extent)
        x,y,v,yaw = x0
        if htype=="constvel":
            vx = v*np.cos(yaw)
            vy = v*np.sin(yaw)
            ax = np.zeros_like(vx)
            ay = np.zeros_like(vy)
            yaw = yaw*np.ones(T)
            scene_ts = np.arange(0,T)
            x = x+vx*(scene_ts-T+1)*dt
            y = y+vy*(scene_ts-T+1)*dt
        elif htype in ["brake","accelerate"]:
            acce = -2.0 if htype=="brake" else 2.0
            seq = np.concatenate([np.ones(T-int(T/2))*int(T/2),np.arange(int(T/2)-1,-1,-1)])
            vseq = np.clip(v-acce*seq*dt,0.0,10.0)
            vx = vseq*np.cos(yaw)
            vy = vseq*np.sin(yaw)
            ax = vx[1:]-vx[:-1]
            ax = np.concatenate([ax,ax[-1:]])
            ay = vy[1:]-vy[:-1]
            ay = np.concatenate([ay,ay[-1:]])
            yaw = yaw*np.ones(T)
            scene_ts = np.arange(0,T)
            x = x+(vx.cumsum()-vx.sum())*dt
            y = y+(vy.cumsum()-vy.sum())*dt

        track_id = [name]*T
        z = np.zeros(T)
        
        
        pd_frame = pd.DataFrame({"agent_id":track_id,"scene_ts":scene_ts,"x":x,"y":y,"z":z,"heading":yaw,"vx":vx,"vy":vy,"ax":ax,"ay":ay})
        total_agent_data.append(pd_frame)
        total_agent_metadata.append(agent_meta)
    total_agent_data = pd.concat(total_agent_data).set_index(["agent_id", "scene_ts"])
    return total_agent_data,total_agent_metadata


        
def generate_drivesim_scene():
    
    repeat = 10

    dt = 0.1
    data_dir = ""
    T = 20
    cache_path = Path("/home/yuxiaoc/.unified_data_cache")
    env_metadata = EnvMetadata("drivesim",
                               data_dir,
                               dt,
                               parts=[("train",),("main",)],
                               scene_split_map=defaultdict(lambda: "train"),
                               map_locations=("main",))
    env_cache = EnvCache(cache_path)
    scene_records = list()
    
    agent_initial_state = [(np.array([-565,-1001,3.0,0]),"accelerate"),
                           (np.array([-573,-1001,3.0,0]),"accelerate"),
                           (np.array([-573,-1005,3.0,0.0]),"accelerate"),
                           (np.array([-530.0,-976,0.0,-0.95*np.pi/2]),"constvel"),
                           (np.array([-526.4,-976,0.0,-0.95*np.pi/2]),"brake"),
                           (np.array([-507.8,-1027,0.0,np.pi/2+0.08]),"brake"),
                           (np.array([-507.8,-1021,0.0,np.pi/2+0.08]),"brake"),
                           (np.array([-504,-1021,0.0,np.pi/2+0.08]),"brake"),
                           (np.array([-500.5,-1021,0.0,np.pi/2+0.08]),"brake"),
                           (np.array([-480.5,-992,0.0,np.pi]),"brake"),
                           (np.array([-473,-992,0.0,np.pi]),"brake"),
                           (np.array([-489,-1049,8.0,np.pi/2-0.1]),"constvel"),
                           (np.array([-488.4,-983,5.0,np.pi*0.75]),"accelerate"),
                           (np.array([-524.2,-964.6,2.0,-0.95*np.pi/2]),"brake"),
                           (np.array([-523.3,-976,0.0,-0.95*np.pi/2]),"brake"),
                           (np.array([-519.5,-975,0.0,-0.92*np.pi/2]),"brake"),
                           (np.array([-563.4,-1011,8.0,-0.18*np.pi/2]),"constvel"),
                            ]

    noise_spec = [1,1,1,2,2,3,3,3]
    for r in range(repeat):
        group1_xn =np.random.randn()*5
        group1_yn = np.random.randn()*0.5
        group1_vn = np.random.randn()*0.5
        group1_psin = np.random.randn()*0.01
        group2_xn = np.random.randn()*0.2
        group2_yn = np.random.randn()*0.2
        group2_vn = 0
        group2_psin = np.random.randn()*0.01
        group3_xn = np.random.randn()*0.2
        group3_yn = np.random.randn()*0.2
        group3_vn = 0
        group3_psin = np.random.randn()*0.01
        group1_n = np.array([group1_xn,group1_yn,group1_vn,group1_psin])
        group2_n = np.array([group2_xn,group2_yn,group2_vn,group2_psin])
        group3_n = np.array([group3_xn,group3_yn,group3_vn,group3_psin])
        init_state=np.stack([x0 for x0,_ in agent_initial_state])
        hist_types = [htype for _,htype in agent_initial_state]
        for i in range(len(init_state)):
            if i <len(noise_spec):
                if noise_spec[i] == 1:
                    init_state[i]+=group1_n
                elif noise_spec[i] == 2:
                    init_state[i]+=group2_n
                elif noise_spec[i] == 3:
                    init_state[i]+=group3_n
        
        agent_names =["ego"]+[str(i) for i in range(769,769+len(agent_initial_state)-1)]
        agent_extents = [FixedExtent(length=4.0,width=2.0,height=2.0)]*len(agent_names)
        
        total_agent_data,total_agent_metadata = generate_agentmeta(init_state,agent_names,agent_extents,T,dt,hist_types)
        Scene_main = Scene(env_metadata,
                        name= f"scene_{r}",
                        location="main",
                        data_split = "train",
                        length_timesteps=T,
                        raw_data_idx=0,
                        data_access_info=None,
                        description = None,
                        agents = total_agent_metadata,
                        agent_presence=[total_agent_metadata]*T,)
        
        DataFrameCache.save_agent_data(total_agent_data, cache_path, Scene_main)
    
        env_cache.save_scene(Scene_main)
        scene_r_rec = DrivesimSceneRecord(name=f"scene_{r}",
                                        location = "main",
                                        length = T,
                                        split = "train",
                                        # desc: str
                                        data_idx=0)
        scene_records.append(scene_r_rec)
    
    with open(cache_path/"drivesim"/"scenes_list.dill", "wb") as f:
        dill.dump(scene_records, f)
    visualize_scene(agent_initial_state)

def plot_lane(lane,plot,color="grey"):
    bdry_l,_ = LaneUtils.get_edge(lane,dir="L",num_pts=15)
    bdry_r,_ = LaneUtils.get_edge(lane,dir="R",num_pts=15)
    lane_center,_ = LaneUtils.get_edge(lane,dir="C",num_pts=15)
    bdry_xy = np.concatenate([bdry_l,np.flip(bdry_r,0)],0)
    patch_glyph = plot.patch(x=bdry_xy[:,0],y=bdry_xy[:,1],fill_alpha=0.5,color = color)
    centerline_glyph = plot.line(x=lane_center[:,0],y=lane_center[:,1],line_dash="dashed",line_width=2)
    return patch_glyph,centerline_glyph
def get_agent_edge(xy,h,extent):
    
    edges = np.array([[0.5,0.5],[0.5,-0.5],[-0.5,-0.5],[-0.5,0.5]])*extent[np.newaxis,:2]
    rotM = np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
    edges = (rotM@edges[...,np.newaxis]).squeeze(-1)+xy[np.newaxis,:]
    return edges
def visualize_scene(agent_initial_state):
    plot = figure(name='base',height=1000, width=1000, title="traffic Animation",  
                tools="reset,save,pan,hover,wheel_zoom",toolbar_location="below",match_aspect=True)
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None  
    plot.axis.visible=False
    cache_path = Path("~/.unified_data_cache").expanduser()
    mapAPI = MapAPI(cache_path)
    map_name = "drivesim:main"
    vec_map = mapAPI.get_map(map_name, scene_cache=None)
    xyz = np.array([-500,-1000,0])
    lanes_t = vec_map.get_lanes_within(xyz,60)
    lanecenter_glyph = dict()
    lanes = set()
    lanepatch_glyph = dict()
    for lane in lanes_t:
        patch_glyph,centerline_glyph = plot_lane(lane,plot)
        lanecenter_glyph[lane] = centerline_glyph
        lanepatch_glyph[lane] = patch_glyph
        lanes.add(lane)
    extent = np.array([4.5,2.5])
    agent_patch = dict()
    palette = bokeh.palettes.Category20[20]
    agent_color = ["blueviolet"] + [palette[i%20] for i in range(len(agent_initial_state)-1)]
    # for i in range(len(agent_initial_state)):
    #     edges = get_agent_edge(agent_initial_state[i][0][:2],agent_initial_state[i][0][3],extent)
    #     agent_patch[i] = plot.patch(x=edges[:,0],y=edges[:,1],color=agent_color[i])
    save(plot)
    show(plot)
if __name__ == "__main__":
    generate_drivesim_scene()
