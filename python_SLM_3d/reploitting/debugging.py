from pathlib import Path

p = Path("/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/")
for f in p.glob("*.bmp"):
    print(f.name)
