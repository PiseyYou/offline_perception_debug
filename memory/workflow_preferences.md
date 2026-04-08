---
name: Project Workflow Preferences
description: How the user prefers to integrate Claude skills and handle planning
type: feedback
---

## Skill Integration Rule
Always use `planning-with-files` for multi-step tasks. Create a `task_plan.md` in the root directory to track progress.

**Why:** The user wants to see clear footprints of the AI's logic and maintain continuity across sessions.

**How to apply:** Before writing any code for a complex task, invoke `writing-plans` to generate the file-based plan.