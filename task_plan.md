# Task Plan: Unified Perception Processor & Debug Logic Merge

## Goal
Merge `offline_perception_debug.cpp` and `offline_perception_debug_432_sob.cpp` logic into `UnifiedPerceptionProcessor`. Prioritize 432_sob logic, support configurable image sizes (640x384 or 640x480), and consolidate Mode 0-8 logic.

## Phases

### Phase 1: Update Configuration System
- [ ] Update `Config` struct in `unified_perception_processor.h` to include image dimensions (width/height)
- [ ] Add logic to load these dimensions from `config.yaml` or a config parser
- **Status**: in_progress

### Phase 2: Merge Processing Logic (Prioritize 432_sob)
- [ ] Copy and adapt utility functions (refineObstacle, computeStraightEdgeScore, etc.) from 432_sob into `UnifiedPerceptionProcessor`
- [ ] Implement Mode 0-8 processing logic in `src/unified_perception_processor.cpp` following the 432_sob pattern
- [ ] Ensure all modes respect the configurable image size
- **Status**: pending

### Phase 3: Integration and Cleanup
- [ ] Update `src/offline_perception_debug.cpp` to use the new `UnifiedPerceptionProcessor` as the single entry point
- [ ] Remove redundant `.cpp` files if possible
- [ ] Verify build and basic functionality
- **Status**: pending

## Decisions
- Use `offline_perception_debug_432_sob.cpp` as the source of truth for perception logic.
- Support dynamic resizing based on config or model requirements.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|