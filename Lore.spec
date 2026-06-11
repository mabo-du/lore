# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os
import site

# Manually find CTranslate2 binaries
# Assuming site-packages is in the venv
site_packages = site.getsitepackages()[0]
ctranslate2_lib = os.path.join(site_packages, 'ctranslate2', 'lib')

binaries = []
if os.path.exists(ctranslate2_lib):
    # Package all .so files from ctranslate2/lib
    binaries.append((os.path.join(ctranslate2_lib, '*.so*'), 'ctranslate2/lib'))
else:
    print(f"Warning: ctranslate2 lib not found at {ctranslate2_lib}")

hiddenimports = [
    'PyQt6.QtQuick',
    'requests',
    'platformdirs',
    'urllib3',
    'pyannote.audio',
    'resemblyzer',
    'sklearn.cluster',
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',
    'cryptography',
    'cryptography.hazmat',
    'cryptography.hazmat.primitives',
    'cryptography.hazmat.primitives.kdf',
    'cryptography.hazmat.primitives.kdf.pbkdf2',
    'cryptography.fernet',
    'utils.token_vault',
]

datas = []
# PyQt6 plugins will be handled automatically by modern PyInstaller, 
# but imageio-ffmpeg executable needs to be bundled if used.
# Modern imageio-ffmpeg automatically bundles its binary.

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorboard', 'tensorboardX'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Lore',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Lore',
)
