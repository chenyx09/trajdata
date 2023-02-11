#!/usr/bin/env python

# Copyright 2021 daohu527 <daohu527@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

if typing.TYPE_CHECKING:
    from imap.lib.opendrive.junction import Junction

import math
from typing import Optional

import imap.global_var as global_var
from imap.lib.opendrive.common import convert_speed
from imap.lib.opendrive.lanes import Lanes
from imap.lib.opendrive.plan_view import PlanView
from imap.lib.opendrive.profile import ElevationProfile, LateralProfile
from imap.lib.opendrive.road_objects import RoadObjects
from imap.lib.opendrive.signals import Signals
from intervaltree import IntervalTree


# Type
class Speed:
    def __init__(self, max_speed=None, unit=None):
        self.max_speed = max_speed
        self.unit = unit

    def parse_from(self, raw_speed):
        if raw_speed is None:
            return
        raw_max_speed = raw_speed.attrib.get("max")
        self.unit = raw_speed.attrib.get("unit")
        if raw_max_speed == "no limit" or raw_max_speed == "undefined":
            self.max_speed = 0
        else:
            self.max_speed = convert_speed(raw_max_speed, self.unit)


class RoadType:
    def __init__(self, s=None, road_type=None):
        self.s = s
        self.road_type = road_type

    def add_speed(self, speed):
        self.speed = speed

    def parse_from(self, raw_road_type):
        if raw_road_type is not None:
            self.s = raw_road_type.attrib.get("s")
            self.road_type = raw_road_type.attrib.get("type")

            raw_speed = raw_road_type.find("speed")
            speed = Speed()
            speed.parse_from(raw_speed)
            self.add_speed(speed)


# Link
class RoadLink:
    def __init__(self, element_type=None, element_id=None, contact_point=None):
        self.element_type = element_type
        self.element_id = element_id
        self.contact_point = contact_point

    def parse_from(self, raw_data):
        if raw_data is not None:
            self.element_type = raw_data.attrib.get("elementType")
            self.element_id = raw_data.attrib.get("elementId")
            self.contact_point = raw_data.attrib.get("contactPoint")


class Link:
    def __init__(self, predecessor=None, successor=None):
        self.predecessor = RoadLink()
        self.successor = RoadLink()

        # private
        self.predecessor_road: Optional[Road] = None
        self.successor_road: Optional[Road] = None
        self.predecessor_junction: Optional[Junction] = None
        self.successor_junction: Optional[Junction] = None

    def parse_from(self, raw_link):
        if raw_link is not None:
            raw_predecessor = raw_link.find("predecessor")
            self.predecessor.parse_from(raw_predecessor)

            raw_successor = raw_link.find("successor")
            self.successor.parse_from(raw_successor)


# Road
class Road:
    def __init__(self, name=None, length=None, road_id=None, junction_id=None):
        self.name = name
        self.length = length
        self.road_id = road_id
        self.junction_id = junction_id

        self.link = Link()
        self.road_type = RoadType()
        self.plan_view = PlanView()
        self.elevation_profile = ElevationProfile()
        self.lateral_profile = LateralProfile()
        self.lanes = Lanes()
        self.signals = Signals()
        self.objects = RoadObjects()

        # private
        self.reference_line = IntervalTree()

    def get_xyz_at_sth(self, s: float, t: float, h: float):
        return self.plan_view.get_xyz_at_sthe(s, t, h, self.elevation_profile)

    def generate_lane_boundary(self):
        for lane_section in self.lanes.lane_sections:
            lane_section.left
            lane_section.right

    def post_processing(self):
        # add length
        for idx in range(len(self.lanes.lane_sections) - 1):
            self.lanes.lane_sections[idx].end_s = self.lanes.lane_sections[idx + 1].s
        self.lanes.lane_sections[-1].end_s = self.length

        for idx, lane_section in enumerate(self.lanes.lane_sections):
            length = lane_section.end_s - lane_section.s
            assert length > 0, "Road_{}_Section_{} length is below zero".format(
                self.road_id, idx
            )
            lane_section.set_lane_length(length)

        # add neighbor
        for lane_section in self.lanes.lane_sections:
            lane_section.add_neighbors()

    def parse_from(self, raw_road):
        self.name = raw_road.attrib.get("name")
        self.length = float(raw_road.attrib.get("length"))
        self.road_id = raw_road.attrib.get("id")
        self.junction_id = raw_road.attrib.get("junction")

        raw_link = raw_road.find("link")
        self.link.parse_from(raw_link)

        raw_road_type = raw_road.find("type")
        self.road_type.parse_from(raw_road_type)

        # reference line
        raw_plan_view = raw_road.find("planView")
        assert raw_plan_view is not None, "Road {} has no reference line!".format(
            self.road_id
        )
        self.plan_view.parse_from(raw_plan_view)

        # elevationProfile
        raw_elevation_profile = raw_road.find("elevationProfile")
        self.elevation_profile.parse_from(raw_elevation_profile)

        # lateralProfile
        raw_lateral_profile = raw_road.find("lateralProfile")
        self.lateral_profile.parse_from(raw_lateral_profile)

        # lanes
        raw_lanes = raw_road.find("lanes")
        assert raw_lanes is not None, "Road {} has no lanes!".format(self.road_id)
        self.lanes.parse_from(raw_lanes)

        # objects
        raw_objects = raw_road.find("objects")
        self.objects.parse_from(raw_objects)

        # signals
        raw_signals = raw_road.find("signals")
        self.signals.parse_from(raw_signals)

        # post processing
        self.post_processing()

    def generate_reference_line(self):
        for geometry in self.plan_view.geometrys:
            self.reference_line.addi(geometry.s, geometry.s + geometry.length, geometry)

        assert len(self.reference_line) != 0, "Road {} reference line is empty!".format(
            self.road_id
        )

    def process_lanes(self):
        # generate boundary
        self.lanes.process_lane_sections(self.reference_line, self.elevation_profile)

    def get_cross_section(self, relation):
        return self.lanes.get_cross_section(relation)
