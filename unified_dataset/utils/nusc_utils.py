from nuscenes.nuscenes import NuScenes


def frame_generator(nusc_obj: NuScenes, first_sample_token: str) -> int:
    """Loops through all frames in a scene and yields them for the caller to deal with the information.
    """
    curr_scene_token: str = first_sample_token
    while curr_scene_token:
        frame = nusc_obj.get('sample', curr_scene_token)

        yield frame

        curr_scene_token = frame['next']
