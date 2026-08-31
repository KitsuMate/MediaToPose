# Releasing

Pushing a version tag builds and validates the platform packages, then creates
the GitHub release without manual approval.

Publishing the same packages to Blender Extensions uses the protected GitHub
environment named `blender-extensions`. Configure this environment in the
repository settings before the first release:

1. Add at least one required reviewer.
2. Add `BLENDER_EXTENSIONS_TOKEN` as an environment secret.
3. Do not add the GitHub release job to this environment.

After the GitHub release is available, GitHub Actions waits for environment
approval before starting the Blender Extensions publishing job.
