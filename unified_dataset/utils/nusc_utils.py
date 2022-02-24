from nuscenes.nuscenes import NuScenes


def traverse_scene_get(nusc_obj: NuScenes, first_sample_token: str) -> int:
    # Placeholder function for when I need to traverse a scene and get a specific value from each frame.
    scene_length: int = 0
    curr_scene_token: str = first_sample_token
    while curr_scene_token:
        scene_length += 1
        frame = nusc_obj.get('sample', curr_scene_token)
        curr_scene_token = frame['next']

    return scene_length