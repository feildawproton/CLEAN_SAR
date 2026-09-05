from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CleanPhysicsConfig:
    """
    Plain scalar parameters defining SAR image geometry, sample spacing, and bandwidth.
    
    Decoupled from file formats, SARkit, and XML trees.
    Directly corresponds to a C/CUDA plain-old-data struct for native acceleration.
    """
    row_ss: float
    col_ss: float
    row_bw: float
    col_bw: float
    row_wid: float
    col_wid: float
    scp_slant_range: float
    scp_row: float
    scp_col: float
    chip_start_row: int = 0
    chip_start_col: int = 0
    row_wgt: str = "UNIFORM"
    col_wgt: str = "UNIFORM"

    @classmethod
    def from_sicd_handler(cls, handler, chip_start: Tuple[int, int] = (0, 0)):
        """Builds configuration from a SICDHandler instance."""
        return cls(
            row_ss=float(handler.row_ss),
            col_ss=float(handler.col_ss),
            row_bw=float(handler.row_bw),
            col_bw=float(handler.col_bw),
            row_wid=float(handler.row_wid),
            col_wid=float(handler.col_wid),
            scp_slant_range=float(handler.scp_slant_range),
            scp_row=float(handler.scp_pixel[0]),
            scp_col=float(handler.scp_pixel[1]),
            chip_start_row=int(chip_start[0]),
            chip_start_col=int(chip_start[1]),
            row_wgt=str(handler.row_wgt_name or "UNIFORM"),
            col_wgt=str(handler.col_wgt_name or "UNIFORM"),
        )

    def global_to_metric(self, r_global: float, c_global: float) -> Tuple[float, float]:
        """Converts global pixel indices to SCP-centered metric coordinates (meters)."""
        x_row = (r_global - self.scp_row) * self.row_ss
        y_col = (c_global - self.scp_col) * self.col_ss
        return x_row, y_col

    def chip_to_global(self, r_chip: float, c_chip: float) -> Tuple[float, float]:
        """Converts local chip coordinates to global pixel indices."""
        return r_chip + self.chip_start_row, c_chip + self.chip_start_col
