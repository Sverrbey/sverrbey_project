import os
import glob
from satpy import Scene
import warnings
warnings.filterwarnings("ignore")

channels = ["vis_04", "vis_05", "vis_06", "vis_08", "vis_09", "nir_13", "nir_16", "nir_22", "ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105", "ir_123", "ir_133"]
rgb                         = "natural_color"
fcil1c_data_archive_path    = "data/fci_data"
exports_path                = "data/fci_exports"
repeat_cycle                = "0073"
data_reader                 = "fci_l1c_nc"      
# create the scene
scn = Scene(filenames=glob.glob(os.path.join(fcil1c_data_archive_path, f"W*FDHSI*BODY*{repeat_cycle}_????.nc")), reader=data_reader)

# radiances export
for channel in channels:
    scn.load([channel], upper_right_corner="NE")
    export_path = os.path.join(exports_path, "radiances", f"RC{repeat_cycle}_{channel}.png")
    scn.save_dataset(channel, calibration="radiance", filename=export_path)
    print(f"Radiances for channel {channel} exported to {export_path}.")

# reflectances or brightness temperatures export
for channel in channels:
    scn.load([channel], upper_right_corner="NE")
    export_path = os.path.join(exports_path, "reflectances_or_brightness_temperatures", f"RC{repeat_cycle}_{channel}.png")
    scn.save_dataset(channel, filename=export_path)
    print(f"Reflectances/brightness temperatures for channel {channel} exported to {export_path}.")


# RGB export
scn.load([rgb], upper_right_corner="NE")
export_path = os.path.join(exports_path, "rgbs", f"RC{repeat_cycle}_{rgb}.png")
scn.save_dataset(rgb, filename=export_path)
print(f"RGB exported to {export_path}.")