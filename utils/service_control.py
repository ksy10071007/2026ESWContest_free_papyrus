"""Safe argv construction for administrator service controls."""

from pathlib import Path


ALLOWED_ADMIN_ACTIONS = frozenset({'restart', 'stop'})


def service_manager_argv(project_root, action):
    if action not in ALLOWED_ADMIN_ACTIONS:
        raise ValueError(f'Unsupported administrator service action: {action}')
    manager_path = (Path(project_root) / 'scripts' / 'mediflow-kiosk').resolve()
    if not manager_path.is_file():
        raise FileNotFoundError(f'Service manager not found: {manager_path}')
    return [str(manager_path), action]
