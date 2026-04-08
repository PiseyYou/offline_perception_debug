# Findings: ROS2 Cleanup

## 2026-04-02
- `perception_common.h` is widely used across the project and must be replaced with a local `perception.h` to ensure offline compilation.
- Found redefinition issues when both headers were present; resolution is to unify on `perception.h`.