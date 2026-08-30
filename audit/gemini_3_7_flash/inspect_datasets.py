import os
import glob
import numpy as np
import lxml.etree as etree
import sarkit.sicd as ss

def inspect_all():
    data_files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    diffpfa_files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    
    all_files = [("data", f) for f in data_files] + [("diffpfa", f) for f in diffpfa_files]
    
    print(f"Total SICD NITF files found: {len(all_files)}")
    print("=" * 80)
    
    for tag, filepath in all_files:
        print(f"\n[{tag.upper()}] File: {os.path.basename(filepath)} ({os.path.getsize(filepath) / (1024*1024):.2f} MB)")
        with open(filepath, "rb") as fp, ss.NitfReader(fp) as reader:
            meta = reader.metadata
            xml = meta.xmltree
            xh = ss.XmlHelper(xml)
            
            num_rows = xh.load("./{*}ImageData/{*}NumRows")
            num_cols = xh.load("./{*}ImageData/{*}NumCols")
            scp_pix = xh.load("./{*}ImageData/{*}SCPPixel")
            
            row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
            col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
            row_bw = xh.load("./{*}Grid/{*}Row/{*}ImpRespBW")
            col_bw = xh.load("./{*}Grid/{*}Col/{*}ImpRespBW")
            row_wid = xh.load("./{*}Grid/{*}Row/{*}ImpRespWid")
            col_wid = xh.load("./{*}Grid/{*}Col/{*}ImpRespWid")
            
            row_wgt = xh.load("./{*}Grid/{*}Row/{*}WgtType/{*}WindowName")
            col_wgt = xh.load("./{*}Grid/{*}Col/{*}WgtType/{*}WindowName")
            
            image_type = xh.load("./{*}Grid/{*}Type")
            pfa_node = xml.find("{*}PFA")
            rma_node = xml.find("{*}RMA")
            
            print(f"  Dimensions: Rows={num_rows}, Cols={num_cols}")
            print(f"  SCP Pixel: {scp_pix}")
            print(f"  Grid Type: {image_type}")
            print(f"  Sample Spacing (m): Row={row_ss}, Col={col_ss}")
            print(f"  Impulse Resp BW (1/m): Row={row_bw}, Col={col_bw}")
            print(f"  Impulse Resp Wid (m): Row={row_wid}, Col={col_wid}")
            print(f"  Weighting Windows: Row={row_wgt}, Col={col_wgt}")
            print(f"  Formation Model: PFA={pfa_node is not None}, RMA={rma_node is not None}")
            
            if pfa_node is not None:
                krg1 = xh.load("./{*}PFA/{*}Krg1")
                krg2 = xh.load("./{*}PFA/{*}Krg2")
                kaz1 = xh.load("./{*}PFA/{*}Kaz1")
                kaz2 = xh.load("./{*}PFA/{*}Kaz2")
                pa_poly = xh.load("./{*}PFA/{*}PolarAngPoly")
                sf_poly = xh.load("./{*}PFA/{*}SpatialFreqSFPoly")
                print(f"  PFA Krg: [{krg1}, {krg2}], Kaz: [{kaz1}, {kaz2}]")
                print(f"  PFA PolarAngPoly: {pa_poly}")
                print(f"  PFA SpatialFreqSFPoly: {sf_poly}")

if __name__ == "__main__":
    inspect_all()
