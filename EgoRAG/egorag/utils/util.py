def time_to_frame_idx(time_int: int, fps: int) -> int:
    """
    Convert time in HHMMSSFF format (integer or string) to frame index.
    :param time_int: Time in HHMMSSFF format, e.g., 10483000 (10:48:30.00) or "10483000".
    :param fps: Frames per second of the video.
    :return: Frame index corresponding to the given time.
    """
    # Ensure time_int is a string for slicing
    time_str = str(time_int).zfill(
        8
    )  # Pad with zeros if necessary to ensure it's 8 digits

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    frames = int(time_str[6:8])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = total_seconds * fps + frames  # Convert to total frames

    return total_frames
