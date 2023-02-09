from typing import List

import numpy as np


class RoadObject:
    def __init__(self,
        name = None,
        objId = None,
        origin_s = None,
        origin_t = None,
        zOffset = None,
        objType = None,
        hdg = None,
        pitch = None,
        roll = None
    ):
        self.name = name
        self.id = objId
        
        self.origin_s = origin_s
        self.origin_t = origin_t
        self.zOffset = zOffset
        
        self.type = objType
        
        self.yaw = hdg
        self.pitch = pitch
        self.roll = roll
        
        self.outline_sth: np.ndarray = None
    
    def parse_from(self, raw_object):
        self.name = raw_object.attrib.get("name")
        self.id = raw_object.attrib.get("id")
        
        self.origin_s = float(raw_object.attrib.get("s"))
        if self.origin_s < 0:
            self.origin_s = 0
        
        self.origin_t = float(raw_object.attrib.get("t"))
        self.zOffset = float(raw_object.attrib.get("zOffset"))
        
        self.type = raw_object.attrib.get("type")
        
        self.yaw = float(raw_object.attrib.get("hdg"))
        self.pitch = float(raw_object.attrib.get("pitch"))
        self.roll = float(raw_object.attrib.get("roll"))
        
        raw_outline = raw_object.find("outline")
        
        outline = []
        for corner in raw_outline.iter("cornerLocal"):
            u = float(corner.attrib.get("u"))
            v = float(corner.attrib.get("v"))
            z = float(corner.attrib.get("z"))
            
            outline.append((u, v, z))

        outline_uvz = np.asarray(outline)[..., None]

        # Precompute sines and cosines of Euler angles
        su = np.sin(self.roll)
        cu = np.cos(self.roll)
        sv = np.sin(self.pitch)
        cv = np.cos(self.pitch)
        sw = np.sin(self.yaw)
        cw = np.cos(self.yaw)

        # Create and populate RotationMatrix
        rot_mat = np.zeros((3, 3))
        rot_mat[0, 0] = cv * cw
        rot_mat[0, 1] = su * sv * cw - cu * sw
        rot_mat[0, 2] = su * sw + cu * sv * cw
        rot_mat[1, 0] = cv * sw
        rot_mat[1, 1] = cu * cw + su * sv * sw
        rot_mat[1, 2] = cu * sv * sw - su * cw
        rot_mat[2, 0] = -sv
        rot_mat[2, 1] = su * cv
        rot_mat[2, 2] = cu * cv
                
        outline_sth = rot_mat[None, ...] @ outline_uvz + np.array([self.origin_s, self.origin_t, self.zOffset])[..., None]
        self.outline_sth = outline_sth.squeeze(axis=-1)
        
        # TODO(bivanovic): No idea what's going on here, but crosswalks end up pretty wonky...

class RoadObjects:
    def __init__(self):
        self.objects: List[RoadObject] = []
    
    def parse_from(self, raw_objects):
        if raw_objects is None:
            return
        
        for raw_road_object in raw_objects.iter('object'):
            if raw_road_object.attrib.get("type") == "crosswalk":
                road_obj = RoadObject()
                road_obj.parse_from(raw_road_object)
                self.objects.append(road_obj)
