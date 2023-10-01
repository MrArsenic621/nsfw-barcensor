import launch

if not launch.is_installed("nudenet"):
    launch.run_pip(f"install nudenet", "nudenet")

if not launch.is_installed("diffusers"):
    launch.run_pip(f"install diffusers", "diffusers")