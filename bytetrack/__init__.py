"""ByteTrack multi-object tracker (MIT License, see LICENSE in this folder).

Vendored from https://github.com/FoundationVision/ByteTrack (yolox/tracker,
commit d1bf019) with small changes so it needs only NumPy and SciPy; each
file lists its changes at the top.

    from bytetrack import BYTETracker
    tracker = BYTETracker(SimpleNamespace(track_thresh=0.5, track_buffer=30,
                                          match_thresh=0.8), frame_rate=30)
    tracks = tracker.update(dets)   # dets: (N, 5) x1, y1, x2, y2, score
    for t in tracks: t.track_id, t.tlbr, t.score
"""
from .byte_tracker import BYTETracker, STrack
from .basetrack import TrackState

__all__ = ["BYTETracker", "STrack", "TrackState"]
