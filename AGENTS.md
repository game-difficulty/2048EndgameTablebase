# Repository Instructions

## Production Deployments

Before deploying the main site, live site, or cloud backend, read
`../src - cloud/docs_and_configs/deployment_retention.md` and follow the
deployment retention rules in `../src - cloud/AGENTS.md`.

These rules apply even when deployment is initiated from this local AI repository.

## Desktop Git Directives

Use forward slashes in Windows paths in Codex desktop git directive attributes.

## Local And Cloud Fix Parity

- For bug fixes, check the matching implementation in `../src - cloud`; if the
  same bug exists there, apply and test the corresponding fix in both projects.
- Respect implementation differences and versioned replay compatibility rather
  than copying files blindly. Report which projects were fixed or not affected.
- Edit maintained source only, not historical deployment snapshots or build output.
