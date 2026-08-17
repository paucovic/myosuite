from __future__ import annotations

import inspect
import traceback


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_header("IMPORT MJLAB REGISTRY")

    import myosuite  # noqa: F401
    import mjlab.tasks.registry as registry

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )

    try:
        bootstrap_myosuite_mjlab_registry()
    except Exception:
        traceback.print_exc()

    print_header("CURRENT REGISTERED TASKS")

    try:
        tasks = registry.list_tasks()
        for task in tasks:
            print(task)
    except Exception:
        traceback.print_exc()

    print_header("REGISTRY FUNCTIONS")

    for name in [
        "register_mjlab_task",
        "load_env_cfg",
        "load_rl_cfg",
        "load_runner_cls",
    ]:
        obj = getattr(registry, name, None)
        print()
        print(name, ":", obj)

        if obj is not None:
            try:
                print(inspect.signature(obj))
            except Exception:
                print("Could not inspect signature.")

    print_header("REGISTER_MJLAB_TASK SOURCE")

    try:
        print(inspect.getsource(registry.register_mjlab_task))
    except Exception:
        traceback.print_exc()

    print_header("SEARCH HAND REACH CONFIG IMPORTS")

    candidate_imports = [
        "myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg",
        "myosuite.envs.myo.backends.mjlab.hand_reach_cfg",
        "myosuite.envs.myo.backends.mjlab.configs.handreach_cfg",
    ]

    for module_name in candidate_imports:
        print()
        print("Trying:", module_name)

        try:
            module = __import__(module_name, fromlist=["*"])
            print("SUCCESS:", module)

            names = sorted(dir(module))
            for name in names:
                if "Reach" in name or "reach" in name or "Cfg" in name:
                    value = getattr(module, name)
                    print(" ", name, ":", value)

                    try:
                        print("   signature:", inspect.signature(value))
                    except Exception:
                        pass

        except Exception as error:
            print("FAILED:", repr(error))

    print_header("SEARCH REGISTER_MJLAB_TASK CALLS IN REGISTER FILE")

    try:
        import myosuite.envs.myo.backends.mjlab.register_mjlab_tasks as reg_tasks

        source = inspect.getsource(reg_tasks)

        for line_number, line in enumerate(source.splitlines(), start=1):
            if (
                "register_mjlab_task" in line
                or "myoElbowPose1D6MFixed-v0" in line
                or "HandReach" in line
                or "hand" in line.lower()
            ):
                print(f"{line_number:04d}: {line}")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()