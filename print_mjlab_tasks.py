"""
Print the mjlab tasks that are actually registered.

Run from the repo root:

    python print_mjlab_tasks.py
"""

from __future__ import annotations

import myosuite  # noqa: F401

from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)


def extract_task_ids_from_mjlab_registry() -> list[str]:
    """
    mjlab versions may expose the registry slightly differently.
    This function tries the likely public/internal locations without assuming
    one exact API name.
    """

    import mjlab.tasks.registry as mjlab_registry

    print()
    print("=" * 80)
    print("mjlab.tasks.registry available names")
    print("=" * 80)

    public_names = [name for name in dir(mjlab_registry) if not name.startswith("__")]
    for name in public_names:
        print(name)

    print()
    print("=" * 80)
    print("Searching for registered task dictionary")
    print("=" * 80)

    candidate_names = [
        "TASK_REGISTRY",
        "_TASK_REGISTRY",
        "REGISTRY",
        "_REGISTRY",
        "registry",
        "_registry",
        "TASKS",
        "_TASKS",
    ]

    for name in candidate_names:
        obj = getattr(mjlab_registry, name, None)

        if isinstance(obj, dict):
            print(f"Found dictionary registry: mjlab.tasks.registry.{name}")
            return sorted(str(key) for key in obj.keys())

        if hasattr(obj, "keys"):
            try:
                print(f"Found key-like registry: mjlab.tasks.registry.{name}")
                return sorted(str(key) for key in obj.keys())
            except Exception:
                pass

    candidate_function_names = [
        "list_tasks",
        "list_envs",
        "get_tasks",
        "get_registered_tasks",
    ]

    for name in candidate_function_names:
        fn = getattr(mjlab_registry, name, None)

        if callable(fn):
            print(f"Found registry function: mjlab.tasks.registry.{name}()")
            tasks = fn()

            if isinstance(tasks, dict):
                return sorted(str(key) for key in tasks.keys())

            return sorted(str(task) for task in tasks)

    raise RuntimeError(
        "Could not find mjlab's task registry automatically. "
        "Look at the printed names above and paste them back."
    )


def main() -> None:
    print("=" * 80)
    print("BOOTSTRAPPING MYOSUITE MJLAB REGISTRY")
    print("=" * 80)

    bootstrap_myosuite_mjlab_registry()

    task_ids = extract_task_ids_from_mjlab_registry()

    print()
    print("=" * 80)
    print("REGISTERED MJLAB TASK IDS")
    print("=" * 80)

    relevant_keywords = [
        "myo",
        "elbow",
        "hand",
        "baoding",
        "leg",
        "walk",
    ]

    relevant_task_ids = [
        task_id
        for task_id in task_ids
        if any(keyword in task_id.lower() for keyword in relevant_keywords)
    ]

    if relevant_task_ids:
        for task_id in relevant_task_ids:
            print(task_id)
    else:
        print("No MyoSuite-looking task ids found.")
        print()
        print("All task ids:")
        for task_id in task_ids:
            print(task_id)

    print()
    print("=" * 80)
    print(f"Total registered tasks: {len(task_ids)}")
    print(f"Relevant MyoSuite-looking tasks: {len(relevant_task_ids)}")
    print("=" * 80)


if __name__ == "__main__":
    main()