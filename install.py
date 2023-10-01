import launch

if not launch.is_installed("nudenet"):
    launch.run_pip(f"install nudenet", "nudenet")
