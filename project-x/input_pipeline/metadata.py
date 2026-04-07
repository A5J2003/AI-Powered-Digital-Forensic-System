from pymediainfo import MediaInfo


def extract_metadata(file_path):
    """
    Extracts structured metadata from media file.
    Returns dictionary.
    """

    media_info = MediaInfo.parse(file_path)

    metadata = {}

    for track in media_info.tracks:
        track_data = track.to_data()
        metadata[track.track_type] = track_data

    return metadata