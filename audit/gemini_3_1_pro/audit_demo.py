import os
from clean_sar import SICDHandler, PSFGenerator, run_hogbom_clean

def run_demo():
    input_file = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    output_dir = "/home/feildaw/CLEAN_SAR/audit/gemini_3_1_pro/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "demo_clean.nitf")

    print(f"Loading SICD file: {input_file}")
    handler = SICDHandler(input_file)
    
    # Extract a small chip for quick demonstration
    start_r, start_c = 2777, 6410
    stop_r, stop_c = 2827, 6460 # 50x50 chip
    
    chip, chip_xml = handler.read_chip(start_row=start_r, start_col=start_c, stop_row=stop_r, stop_col=stop_c)
    print(f"Read chip of shape: {chip.shape}")

    psf_gen = PSFGenerator(handler)
    
    print("Running Complex Hogbom CLEAN...")
    result = run_hogbom_clean(
        dirty_image=chip,
        psf_generator=psf_gen,
        method="kspace",
        beam_type="gaussian",
        psf_size=33,
        gain=0.1,
        threshold=0.05,
        max_iters=100,
        chip_origin=(start_r, start_c),
        device="cpu", # Use CPU for quick demo since we just want to test if it runs
        verbose=True,
    )

    print(f"CLEAN converged in {result.iterations} iterations.")
    print(f"Writing result to {output_file}")
    
    handler.write_nitf(output_file, result.clean_image, custom_xmltree=chip_xml)
    print("Demo complete.")

if __name__ == "__main__":
    run_demo()
