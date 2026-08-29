import os
import lxml.etree as etree
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import sarkit.sicd as ss


class SICDHandler:
    """
    Object-oriented handler for NITF SICD complex SAR images and metadata using SARkit.

    Encapsulates:
    - NITF SICD I/O (full image and sub-image chipping)
    - Metadata extraction (Grid, ImageData, PFA/RMA, Projection parameters)
    - Coordinate transformations (chip indices -> global pixel indices -> SCP-relative metric coordinates)
    - Exporting processed images to compliant SICD NITF format.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"SICD file not found: {self.file_path}")

        # Load and parse metadata using sarkit
        with open(self.file_path, "rb") as fp, ss.NitfReader(fp) as reader:
            self.xmltree = etree.ElementTree(etree.fromstring(etree.tostring(reader.metadata.xmltree)))
            self.metadata = reader.metadata
            self.xh = ss.XmlHelper(self.xmltree)

        self._parse_metadata()

    def _safe_load(self, xpath: str, default=None):
        try:
            val = self.xh.load(xpath)
            return val if val is not None else default
        except Exception:
            return default

    def _parse_metadata(self):
        """Extracts essential geometric, grid, and image formation metadata."""
        # ImageData
        self.num_rows = int(self._safe_load("./{*}ImageData/{*}NumRows", 0))
        self.num_cols = int(self._safe_load("./{*}ImageData/{*}NumCols", 0))
        self.first_row = int(self._safe_load("./{*}ImageData/{*}FirstRow", 0))
        self.first_col = int(self._safe_load("./{*}ImageData/{*}FirstCol", 0))

        scp_pix = self._safe_load("./{*}ImageData/{*}SCPPixel", None)
        if scp_pix is not None:
            self.scp_pixel = np.asarray(scp_pix, dtype=np.float64)
        else:
            self.scp_pixel = np.array([self.num_rows // 2, self.num_cols // 2], dtype=np.float64)

        # Grid parameters
        self.row_ss = float(self._safe_load("./{*}Grid/{*}Row/{*}SS", 1.0))
        self.col_ss = float(self._safe_load("./{*}Grid/{*}Col/{*}SS", 1.0))

        self.row_bw = float(self._safe_load("./{*}Grid/{*}Row/{*}ImpRespBW", 1.0 / self.row_ss))
        self.col_bw = float(self._safe_load("./{*}Grid/{*}Col/{*}ImpRespBW", 1.0 / self.col_ss))

        self.row_wid = float(self._safe_load("./{*}Grid/{*}Row/{*}ImpRespWid", 0.886 / self.row_bw))
        self.col_wid = float(self._safe_load("./{*}Grid/{*}Col/{*}ImpRespWid", 0.886 / self.col_bw))

        # Weighting window info
        self.row_wgt_name = self._safe_load("./{*}Grid/{*}Row/{*}WgtType/{*}WindowName", None)
        self.col_wgt_name = self._safe_load("./{*}Grid/{*}Col/{*}WgtType/{*}WindowName", None)

        # PFA metadata (if available)
        self.is_pfa = self.xmltree.find("{*}PFA") is not None
        self.pfa_meta: Dict[str, Any] = {}

        if self.is_pfa:
            self.pfa_meta["Krg1"] = float(self._safe_load("./{*}PFA/{*}Krg1", 0.0))
            self.pfa_meta["Krg2"] = float(self._safe_load("./{*}PFA/{*}Krg2", self.row_bw))
            self.pfa_meta["Kaz1"] = float(self._safe_load("./{*}PFA/{*}Kaz1", -self.col_bw / 2.0))
            self.pfa_meta["Kaz2"] = float(self._safe_load("./{*}PFA/{*}Kaz2", self.col_bw / 2.0))
            self.pfa_meta["PolarAngRefTime"] = float(self._safe_load("./{*}PFA/{*}PolarAngRefTime", 0.0))

            pa_poly = self._safe_load("./{*}PFA/{*}PolarAngPoly", None)
            self.pfa_meta["PolarAngPoly"] = np.asarray(pa_poly if pa_poly is not None else [0.0], dtype=np.float64)

            sf_poly = self._safe_load("./{*}PFA/{*}SpatialFreqSFPoly", None)
            self.pfa_meta["SpatialFreqSFPoly"] = np.asarray(sf_poly if sf_poly is not None else [1.0], dtype=np.float64)

            ipn = self._safe_load("./{*}PFA/{*}IPN", None)
            self.pfa_meta["IPN"] = np.asarray(ipn if ipn is not None else [0.0, 0.0, 1.0], dtype=np.float64)

            fpn = self._safe_load("./{*}PFA/{*}FPN", None)
            self.pfa_meta["FPN"] = np.asarray(fpn if fpn is not None else [0.0, 0.0, 1.0], dtype=np.float64)

        # RMA metadata (if available)
        self.is_rma = self.xmltree.find("{*}RMA") is not None

    def read_full_image(self) -> np.ndarray:
        """
        Reads the full complex image array from the NITF SICD file in native byte order.

        Returns
        -------
        np.ndarray
            2D complex numpy array of shape (num_rows, num_cols), dtype complex64.
        """
        with open(self.file_path, "rb") as fp, ss.NitfReader(fp) as reader:
            img = reader.read_image()
            return np.require(img, dtype=np.complex64, requirements=["C", "A"])

    def read_chip(
        self,
        start_row: int,
        start_col: int,
        stop_row: int,
        stop_col: int,
    ) -> Tuple[np.ndarray, etree.ElementTree]:
        """
        Reads a sub-image (chip / ROI) in native byte order and returns the complex pixel array
        along with an updated SICD XML tree describing the chip.

        Parameters
        ----------
        start_row : int
            Global row index to start reading (inclusive).
        start_col : int
            Global column index to start reading (inclusive).
        stop_row : int
            Global row index to stop reading (exclusive).
        stop_col : int
            Global column index to stop reading (exclusive).

        Returns
        -------
        chip_array : np.ndarray
            2D complex numpy array of shape (stop_row - start_row, stop_col - start_col), dtype complex64.
        chip_xmltree : etree.ElementTree
            Updated SICD XML tree for the sub-image.
        """
        start_row = max(0, min(start_row, self.num_rows - 1))
        start_col = max(0, min(start_col, self.num_cols - 1))
        stop_row = max(start_row + 1, min(stop_row, self.num_rows))
        stop_col = max(start_col + 1, min(stop_col, self.num_cols))

        with open(self.file_path, "rb") as fp, ss.NitfReader(fp) as reader:
            chip_arr, chip_xml = reader.read_sub_image(
                start_row=start_row,
                start_col=start_col,
                stop_row=stop_row,
                stop_col=stop_col,
            )
            chip_native = np.require(chip_arr, dtype=np.complex64, requirements=["C", "A"])
            return chip_native, chip_xml

    def chip_to_global_rowcol(
        self,
        chip_row: Union[int, float, np.ndarray],
        chip_col: Union[int, float, np.ndarray],
        start_row: int,
        start_col: int,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Converts local chip row/col coordinates to global SICD image row/col indices.
        """
        return chip_row + start_row, chip_col + start_col

    def global_to_metric(
        self,
        row: Union[int, float, np.ndarray],
        col: Union[int, float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Converts global pixel indices (row, col) to SCP-centered metric coordinates (xrow, ycol) in meters.
        """
        pts = np.stack([np.asarray(row), np.asarray(col)], axis=-1)
        xy = ss.rowcol_to_xrowycol(self.xmltree, pts)
        return xy[..., 0], xy[..., 1]

    def metric_to_global(
        self,
        xrow: Union[float, np.ndarray],
        ycol: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Converts SCP-centered metric coordinates (xrow, ycol) in meters to global pixel indices (row, col).
        """
        pts = np.stack([np.asarray(xrow), np.asarray(ycol)], axis=-1)
        rc = ss.xrowycol_to_rowcol(self.xmltree, pts)
        return rc[..., 0], rc[..., 1]

    def write_nitf(
        self,
        output_path: str,
        complex_image: np.ndarray,
        custom_xmltree: Optional[etree.ElementTree] = None,
    ):
        """
        Writes a complex SAR image to a NITF SICD file using SARkit.

        Parameters
        ----------
        output_path : str
            Destination filepath.
        complex_image : np.ndarray
            2D complex image array to write.
        custom_xmltree : etree.ElementTree, optional
            SICD XML tree describing the image (e.g. from chip reading). If None, uses original full metadata.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        xml = custom_xmltree if custom_xmltree is not None else self.xmltree

        # Ensure XML image dimensions match the array
        xh = ss.XmlHelper(xml)
        num_rows_elem = xml.find("{*}ImageData/{*}NumRows")
        num_cols_elem = xml.find("{*}ImageData/{*}NumCols")
        if num_rows_elem is not None:
            num_rows_elem.text = str(complex_image.shape[0])
        if num_cols_elem is not None:
            num_cols_elem.text = str(complex_image.shape[1])

        sec = ss.NitfSecurityFields(clas="U")
        meta = ss.NitfMetadata(
            xmltree=xml,
            file_header_part=ss.NitfFileHeaderPart(ostaid="CLEAN_SAR", security=sec),
            im_subheader_part=ss.NitfImSubheaderPart(isorce="CLEAN_SAR_DECONVOLVED", security=sec),
            de_subheader_part=ss.NitfDeSubheaderPart(security=sec),
        )

        with open(output_path, "wb") as fp, ss.NitfWriter(fp, meta) as writer:
            writer.write_image(complex_image.astype(np.complex64))
