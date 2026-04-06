from mujoco_playground._src import mjx_env
print('MENAGERIE_PATH:', mjx_env.MENAGERIE_PATH)
print('Full assets path:', (mjx_env.MENAGERIE_PATH / 'apptronik_apollo' / 'assets').as_posix())

# Check if assets are loaded
from mujoco_playground._src.locomotion.apollo.base import get_assets
assets = get_assets()
print(f'Loaded {len(assets)} assets')
stl_files = [k for k in assets.keys() if k.endswith('.stl')]
print(f'Found {len(stl_files)} STL files:', stl_files[:5])
