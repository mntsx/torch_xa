import sys
from pathlib import Path

# Get the absolute path to the package directory (where __init__.py is)
package_path: str = str(Path(__file__).resolve().parent)

# Add package_path to sys.path (more reliable than PYTHONPATH)
if package_path not in sys.path:
    sys.path.insert(0, package_path)

from src.autograd.engine.interfaces import backward, Superset
from src.autograd.engine.graph import Graph
from src.autograd.configurations.exchangers import XAFexchanger
from src.autograd.configurations.flags import Flag
from src.autograd.configurations.selectors import XAFselector
