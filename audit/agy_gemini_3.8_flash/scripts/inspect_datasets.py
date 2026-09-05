#!/usr/bin/env python3
import glob
import os
import lxml.etree as etree
import sarkit.sicd as ss

files = sorted(glob.glob('/home/feildaw/data/*.nitf') + glob.glob('/home/feildaw/diffpfa/workspace/output/*.nitf'))

schema_1_5_doc = etree.parse('/home/feildaw/CLEAN_SAR/schemas/NGA.STND.0024-4_1.5_Schema.xsd')
schema_1_5 = etree.XMLSchema(schema_1_5_doc)

schema_1_3_doc = etree.parse('/home/feildaw/CLEAN_SAR/schemas/SICD_schema_V1.3.0_2021_11_30.xsd')
schema_1_3 = etree.XMLSchema(schema_1_3_doc)

print(f"Found {len(files)} NITF files.")
print("=" * 100)

for f in files:
    base = os.path.basename(f)
    try:
        with open(f, 'rb') as fp, ss.NitfReader(fp) as reader:
            xml = reader.metadata.xmltree
            root = xml.getroot()
            ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
            
            # Check schema validation
            is_valid_15 = schema_1_5.validate(xml)
            is_valid_13 = schema_1_3.validate(xml)
            
            xh = ss.XmlHelper(xml)
            num_rows = xh.load('./{*}ImageData/{*}NumRows')
            num_cols = xh.load('./{*}ImageData/{*}NumCols')
            first_row = xh.load('./{*}ImageData/{*}FirstRow')
            first_col = xh.load('./{*}ImageData/{*}FirstCol')
            scp_pixel = xh.load('./{*}ImageData/{*}SCPPixel')
            
            row_ss = xh.load('./{*}Grid/{*}Row/{*}SS')
            col_ss = xh.load('./{*}Grid/{*}Col/{*}SS')
            row_bw = xh.load('./{*}Grid/{*}Row/{*}ImpRespBW')
            col_bw = xh.load('./{*}Grid/{*}Col/{*}ImpRespBW')
            row_wid = xh.load('./{*}Grid/{*}Row/{*}ImpRespWid')
            col_wid = xh.load('./{*}Grid/{*}Col/{*}ImpRespWid')
            row_wgt = xh.load('./{*}Grid/{*}Row/{*}WgtType/{*}WindowName')
            col_wgt = xh.load('./{*}Grid/{*}Col/{*}WgtType/{*}WindowName')
            
            slant_range = xh.load('./{*}SCPCOA/{*}SlantRange')
            image_type = xh.load('./{*}ImageFormation/{*}ImageFormationType')
            is_pfa = xml.find('{*}PFA') is not None
            is_rma = xml.find('{*}RMA') is not None
            
            print(f"File: {base}")
            print(f"  Namespace: {ns}")
            print(f"  Valid 1.5: {is_valid_15} | Valid 1.3: {is_valid_13}")
            print(f"  Dimensions: {num_rows} x {num_cols} (First: {first_row}, {first_col}) | SCPPixel: {scp_pixel}")
            print(f"  Grid SS: ({row_ss}, {col_ss}) m | BW: ({row_bw}, {col_bw}) cyc/m | Wid: ({row_wid}, {col_wid}) m")
            print(f"  Grid Wgt: Row='{row_wgt}', Col='{col_wgt}'")
            print(f"  SlantRange: {slant_range} m | Formation: {image_type} | PFA: {is_pfa} | RMA: {is_rma}")
            if not is_valid_15:
                print("  Schema 1.5 errors (first 3):")
                for err in schema_1_5.error_log[:3]:
                    print(f"    - line {err.line}: {err.message}")
            print("-" * 100)
    except Exception as e:
        print(f"Error reading {base}: {e}")
