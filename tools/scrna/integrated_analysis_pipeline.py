import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


def log(msg):
    print(f"\n{'='*60}\n[Pipeline] {msg}\n{'='*60}\n", flush=True)


def run_command(cmd_list, step_name):
    cmd_str = " ".join([str(x) for x in cmd_list])
    log(f"Starting {step_name}...\nCommand: {cmd_str}")
    start_time = time.time()
    try:
        process = subprocess.run(cmd_list, check=True, text=True)
        elapsed = time.time() - start_time
        log(f"{step_name} Completed Successfully! (Time: {elapsed:.2f}s)")
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] {step_name} Failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Single-Cell Analysis & Patient Classification Pipeline"
    )
    parser.add_argument(
        "--input_data", required=True, help="Path to input 10x directory or .h5ad file"
    )
    parser.add_argument("--outdir", required=True, help="Main output directory for the pipeline")
    parser.add_argument(
        "--scmulan_ckpt", required=True, help="Path to scMulan checkpoint directory"
    )
    parser.add_argument(
        "--classifier_model", required=True, help="Path to the trained classifier model (.pt)"
    )
    parser.add_argument(
        "--classifier_info",
        required=True,
        help="Path to model_info.json (generated during training, contains gene signatures)",
    )
    parser.add_argument(
        "--sample_id", default="MySample", help="Sample ID to use for prediction logging"
    )
    parser.add_argument("--script_step1", default="1.scrna_preannotation.py")
    parser.add_argument("--script_step2", default="2.cluster_auto_annotation.py")
    parser.add_argument("--script_step3", default="3.annotate_and_split.py")
    parser.add_argument("--script_step4", default="patient_cluster_classifier_predict.py")
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    scripts_dir = Path(__file__).parent.resolve()
    api_config_path = os.environ.get(
        "BIOPLEX_API_CONFIG", str((scripts_dir / "api_config.json").resolve())
    )
    dir_step1 = outdir / "01_PreAnnotation"
    dir_step2 = outdir / "02_AutoAnnotation"
    dir_step3 = outdir / "03_FinalResult"
    cmd_step1 = [python_exe, scripts_dir / args.script_step1, "--outdir", str(dir_step1)]
    input_path = Path(args.input_data)
    if input_path.is_dir() and not input_path.name.endswith(".h5ad"):
        cmd_step1.extend(["--input_10x_parent", str(input_path)])
    else:
        cmd_step1.extend(["--input_h5ad", str(input_path)])
    run_command(cmd_step1, "Step 1: Pre-annotation")
    step1_h5ad = dir_step1 / "final_preannotation.h5ad"
    if not step1_h5ad.exists():
        log(f"[ERROR] Step 1 did not generate {step1_h5ad}")
        sys.exit(1)
    dir_step2.mkdir(exist_ok=True)
    step2_csv = dir_step2 / "cluster_annotations.csv"
    cmd_step2 = [
        python_exe,
        scripts_dir / args.script_step2,
        "--adata_h5ad",
        str(step1_h5ad),
        "--ckpt_path",
        args.scmulan_ckpt,
        "--out_csv",
        str(step2_csv),
        "--cluster_key",
        "leiden",
        "--celltype_key",
        "cell_type_pred",
        "--run_tumor_analysis",
        "--api_config_file",
        api_config_path,
    ]
    run_command(cmd_step2, "Step 2: Auto-annotation")
    if not step2_csv.exists():
        log(f"[ERROR] Step 2 did not generate {step2_csv}")
        sys.exit(1)
    final_cell_type_key = "cell_type_final"
    cmd_step3 = [
        python_exe,
        scripts_dir / args.script_step3,
        "--adata_h5ad",
        str(step1_h5ad),
        "--csv",
        str(step2_csv),
        "--outdir",
        str(dir_step3),
        "--cell_type_key",
        final_cell_type_key,
    ]
    run_command(cmd_step3, "Step 3: Annotate, Plot & Split")
    step3_h5ad = dir_step3 / "annotated_complete.h5ad"
    if not step3_h5ad.exists():
        log(f"[ERROR] Step 3 did not generate {step3_h5ad}")
        sys.exit(1)
    cmd_step4 = [
        python_exe,
        scripts_dir / args.script_step4,
        "predict_one",
        "--model",
        args.classifier_model,
        "--h5ad",
        str(step3_h5ad),
        "--sc_model_info",
        args.classifier_info,
        "--sample_id",
        args.sample_id,
        "--cell_type_key",
        final_cell_type_key,
    ]
    run_command(cmd_step4, "Step 4: Patient Classification Prediction")
    log(
        f"All steps completed successfully!\nFinal H5AD: {step3_h5ad}\nCheck console output for Step 4 prediction results."
    )


if __name__ == "__main__":
    main()
