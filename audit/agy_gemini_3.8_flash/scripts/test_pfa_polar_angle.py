import glob
import os
import numpy as np
import sarkit.sicd as ss

files = sorted(glob.glob('/home/feildaw/data/*.nitf') + glob.glob('/home/feildaw/diffpfa/workspace/output/*.nitf'))[:3]

for f in files:
    base = os.path.basename(f)
    with open(f, 'rb') as fp, ss.NitfReader(fp) as reader:
        xml = reader.metadata.xmltree
        xh = ss.XmlHelper(xml)
        
        is_pfa = xml.find('{*}PFA') is not None
        if not is_pfa:
            print(f"{base}: Not PFA")
            continue
            
        num_rows = int(xh.load('./{*}ImageData/{*}NumRows'))
        num_cols = int(xh.load('./{*}ImageData/{*}NumCols'))
        row_ss = float(xh.load('./{*}Grid/{*}Row/{*}SS'))
        col_ss = float(xh.load('./{*}Grid/{*}Col/{*}SS'))
        scp_slant_range = float(xh.load('./{*}SCPCOA/{*}SlantRange'))
        scp_pix = np.asarray(xh.load('./{*}ImageData/{*}SCPPixel'), dtype=np.float64)
        
        t_coa_poly = xh.load('./{*}Grid/{*}TimeCOAPoly')
        pa_poly = xh.load('./{*}PFA/{*}PolarAngPoly')
        
        print(f"\nFile: {base} (Dims: {num_rows}x{num_cols})")
        print(f"  SlantRange R0: {scp_slant_range:.2f} m | SCP Pixel: {scp_pix}")
        print(f"  TimeCOAPoly: {t_coa_poly}")
        print(f"  PolarAngPoly: {pa_poly}")
        
        test_points = [
            ("SCP", scp_pix[0], scp_pix[1]),
            ("Top-Left (0, 0)", 0, 0),
            ("Top-Right (0, W)", 0, num_cols),
            ("Bottom-Left (H, 0)", num_rows, 0),
            ("Bottom-Right (H, W)", num_rows, num_cols),
        ]
        
        print(f"  {'Point':<20} | {'ycol (m)':<10} | {'theta_approx (deg)':<18} | {'theta_poly (deg)':<18} | {'Diff (deg)':<12}")
        print("  " + "-" * 85)
        for name, r, c in test_points:
            xrow = (r - scp_pix[0]) * row_ss
            ycol = (c - scp_pix[1]) * col_ss
            th_approx = np.arctan2(ycol, scp_slant_range + xrow)
            th_approx_deg = np.rad2deg(th_approx)
            
            th_poly_deg = None
            diff_deg = None
            if t_coa_poly is not None and pa_poly is not None:
                poly_arr = np.array(t_coa_poly)
                t_val = 0.0
                for i in range(poly_arr.shape[0]):
                    for j in range(poly_arr.shape[1]):
                        t_val += poly_arr[i, j] * (xrow ** i) * (ycol ** j)
                
                pa_arr = np.array(pa_poly)
                th_val = 0.0
                for n, c_n in enumerate(pa_arr):
                    th_val += c_n * (t_val ** n)
                th_poly_deg = np.rad2deg(th_val)
                diff_deg = abs(th_approx_deg - th_poly_deg)
                
            p_str = f"{th_poly_deg:10.5f}" if th_poly_deg is not None else "N/A"
            d_str = f"{diff_deg:10.5f}" if diff_deg is not None else "N/A"
            print(f"  {name:<20} | {ycol:<10.1f} | {th_approx_deg:10.5f} deg    | {p_str} deg    | {d_str}")
