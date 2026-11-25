"""drongo — direct georeferencing for drone video (Phase 1)."""
from .camera import CameraModel, Intrinsics
from .frames import FlatGround, GroundModel, LocalFrame
from .geometry import Footprint, FrameGeometry, apply_h, relative_homography, rotation_world_from_cam
from .provider import PoseEstimate, PoseProvider, TelemetryPoseProvider
from .telemetry import DjiSrtReader, TelemetryReader, TelemetryTrack, open_telemetry, write_dji_srt
from .dji_embedded import DjiEmbeddedReader, DjiPacketDecoder, iter_dji_packets
from .types import PoseSample, TelemetrySourceKind, lerp_angle_deg, wrap_deg

__version__ = "0.1.0"

__all__ = [
    "CameraModel", "Intrinsics", "FlatGround", "GroundModel", "LocalFrame",
    "Footprint", "FrameGeometry", "apply_h", "relative_homography", "rotation_world_from_cam",
    "PoseEstimate", "PoseProvider", "TelemetryPoseProvider",
    "DjiSrtReader", "DjiEmbeddedReader", "DjiPacketDecoder", "iter_dji_packets", "TelemetryReader", "TelemetryTrack", "open_telemetry", "write_dji_srt",
    "PoseSample", "TelemetrySourceKind", "lerp_angle_deg", "wrap_deg",
]
