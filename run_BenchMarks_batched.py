import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_model_with_file(model_script, args_list, output_file):
    """Run the model with specific arguments and save output to a file."""
    with open(output_file, 'w') as out_file:
        result = subprocess.run(['python', model_script] + args_list, stdout=out_file, stderr=subprocess.STDOUT)
    
    if result.returncode == 0:
        print(f"{args_list[0]} run successfully")
    else:
        print(f"Error running {args_list[0]}")
    
    return result.returncode

def process_files(model_name, input_path, output_path, parallelization=1, gshare_enteries=128, gshare_hist_bits=8, gshare_addr_shift=1):
    """Process the files with the specified model and parallelization level."""
    # Choose the model script based on model name
    model_script = 'model_gshare.py' if model_name == 'gshare' else 'model.py'
    output_suffix = '_gshare.output' if model_name == 'gshare' else '_bht.output'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Check if input path is a file or directory
    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        files_to_process = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        sys.exit(1)

    # Prepare tasks for parallel execution
    tasks = []
    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_path, f"{file_name}{output_suffix}")
        
        # Prepare arguments list based on the model
        if model_script == 'model_gshare.py':
            args_list = [file_path, str(gshare_enteries), str(gshare_hist_bits), str(gshare_addr_shift)]
        else:
            args_list = [file_path]

        # Add task to the list
        tasks.append((model_script, args_list, output_file))

    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=parallelization) as executor:
        results = [executor.submit(run_model_with_file, *task) for task in tasks]
        for future in results:
            if future.result() != 0:
                print("An error occurred while running one of the files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with specified files.")
    parser.add_argument("input_path", help="Path to the input file or directory")
    parser.add_argument("--model", choices=["gshare", "bht"], default="bht", help="Model to use (default: bht)")
    parser.add_argument("--output_path", default="output_files", help="Directory to save output files (default: output_files)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel executions (default: 1)")
    parser.add_argument("--gshare_enteries", type=int, default=128,help="Number of gshare enteries(default: 128)")
    parser.add_argument("--gshare_hist_bits", type=int, default=8,help="Number of gshare history bits (default: 8)")
    parser.add_argument("--gshare_addr_shift", type=int, default=1,help="Number of gshare address shifts(default: 1)")
    
    args = parser.parse_args()

    process_files(args.model, args.input_path, args.output_path, args.parallel, args.gshare_enteries, args.gshare_hist_bits, args.gshare_addr_shift)
