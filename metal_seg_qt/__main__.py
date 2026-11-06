"""Support both module execution and running this file directly in an IDE."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metal_seg_qt.app import main
else:
    from .app import main

if __name__ == '__main__':
    raise SystemExit(main())
