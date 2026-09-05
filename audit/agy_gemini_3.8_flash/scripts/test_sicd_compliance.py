import tempfile
import os
import lxml.etree as etree
import sarkit.sicd as ss
from clean_sar.sicd_handler import SICDHandler

file_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
handler = SICDHandler(file_path)

schema_1_5 = etree.XMLSchema(etree.parse('/home/feildaw/CLEAN_SAR/schemas/NGA.STND.0024-4_1.5_Schema.xsd'))
schema_1_3 = etree.XMLSchema(etree.parse('/home/feildaw/CLEAN_SAR/schemas/SICD_schema_V1.3.0_2021_11_30.xsd'))

# 1. Read chip
r0, c0, r1, c1 = 100, 100, 228, 228
chip, chip_xml = handler.read_chip(r0, c0, r1, c1)

xh_orig = handler.xh
xh_chip = ss.XmlHelper(chip_xml)

print("=== CHIP METADATA INSPECTION ===")
print("Original Dimensions:", xh_orig.load("./{*}ImageData/{*}NumRows"), "x", xh_orig.load("./{*}ImageData/{*}NumCols"))
print("Chip Dimensions:    ", xh_chip.load("./{*}ImageData/{*}NumRows"), "x", xh_chip.load("./{*}ImageData/{*}NumCols"))
print("Original FirstRow/Col:", xh_orig.load("./{*}ImageData/{*}FirstRow"), ",", xh_orig.load("./{*}ImageData/{*}FirstCol"))
print("Chip FirstRow/Col:    ", xh_chip.load("./{*}ImageData/{*}FirstRow"), ",", xh_chip.load("./{*}ImageData/{*}FirstCol"))
print("Original SCPPixel:    ", xh_orig.load("./{*}ImageData/{*}SCPPixel"))
print("Chip SCPPixel:        ", xh_chip.load("./{*}ImageData/{*}SCPPixel"))

# Check schema validation of chip_xml
val_13 = schema_1_3.validate(chip_xml)
val_15 = schema_1_5.validate(chip_xml)
print(f"Chip XML valid under 1.3: {val_13}, under 1.5: {val_15}")

# 2. Write NITF and re-open to inspect written file
with tempfile.NamedTemporaryFile(suffix=".nitf", delete=False) as tmp:
    tmp_path = tmp.name

try:
    handler.write_nitf(tmp_path, chip, custom_xmltree=chip_xml)
    print(f"\nWritten NITF file size: {os.path.getsize(tmp_path)} bytes")
    
    with open(tmp_path, "rb") as fp, ss.NitfReader(fp) as reader:
        written_meta = reader.metadata
        written_xml = written_meta.xmltree
        xh_written = ss.XmlHelper(written_xml)
        
        print("=== WRITTEN NITF METADATA INSPECTION ===")
        print("Header OSTAID:     ", reader.header.ostaid)
        print("Image Subheader ISORCE:", reader.image_segments[0].header.isorce)
        print("Written NumRows/Cols:  ", xh_written.load("./{*}ImageData/{*}NumRows"), "x", xh_written.load("./{*}ImageData/{*}NumCols"))
        print("Written FirstRow/Col:  ", xh_written.load("./{*}ImageData/{*}FirstRow"), ",", xh_written.load("./{*}ImageData/{*}FirstCol"))
        print("Written SCPPixel:      ", xh_written.load("./{*}ImageData/{*}SCPPixel"))
        print("Written Valid 1.3:     ", schema_1_3.validate(written_xml))
        print("Written Valid 1.5:     ", schema_1_5.validate(written_xml))
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

