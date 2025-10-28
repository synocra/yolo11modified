import sys, os
# arahkan path ke root proyek 'halo'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ultralytics.nn.modules import coord_att

print("✅ Modul ditemukan:", coord_att.__file__)
