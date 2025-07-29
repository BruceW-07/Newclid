import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse 

from newclid.agent.ddarn import DDARN
from newclid.agent.human_agent import HumanAgent
from newclid.agent.lm import LMAgent
from newclid.api import GeometricSolverBuilder


def solve_problem(args):
    """
    Process a single problem and return whether it was solved successfully along with the time taken.
    """
    problem_name, problems_path, model_path, decoding_size, beam_size, search_depth = args
    start_time = time.time()
    try:
        solver = (
            GeometricSolverBuilder()
            .load_problem_from_file(problems_path, problem_name)#, rename=True)
            .with_deductive_agent(LMAgent(model_path, decoding_size=decoding_size,beam_size=beam_size, search_depth=search_depth))
            .build()
        )
        is_solved = solver.run()
        elapsed_time = time.time() - start_time
        return (problem_name, is_solved, elapsed_time) 
    except Exception as e:
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (problem_name, False, elapsed_time)

def run_newclid(filepath: Path, modelpath: Path, max_workers: int = 4, decoding_size: int = 1, beam_size: int = 1, search_depth: int = 1):
    """
    Main function, read the file and execute tasks using ProcessPoolExecutor.
    
    Parameters:
        filepath (Path): The path of the file containing problem names.
        max_workers (int): The maximum number of processes in the pool, default is 4.
    """

    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    # Read all problem names (every other line starting from index 0)
    problem_names = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            problem_names.append(lines[i].strip())

    total_problems = len(problem_names)
    print(f"Total problems to solve: {total_problems}")

    # Use ProcessPoolExecutor to process problems concurrently
    solved_count = 0
    processed_count = 0  
    total_time = 0 
    total_real_time = time.time()   

    if max_workers == 1:
        # Single-threaded execution
        for problem_name in problem_names:
            problem_name, is_solved, elapsed_time = solve_problem((problem_name, filepath, modelpath, decoding_size, beam_size, search_depth))
            solved_count += 1 if is_solved else 0
            processed_count += 1  
            total_time += elapsed_time 
            print(
                f"Progress: {processed_count}/{total_problems} processed, "  
                f"Solved: {solved_count}, "
                f"Current: {problem_name} "
                f"({'Success' if is_solved else 'Failed'}), "
                f"Time: {elapsed_time:.2f}s"
            )
    else:
        # Multi-threaded execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and collect futures
            futures = {executor.submit(solve_problem, (name, filepath, modelpath, decoding_size, beam_size, search_depth)): name for name in problem_names}

            # Process completed tasks
            for future in as_completed(futures):
                problem_name = futures[future]
                problem_name, is_solved, elapsed_time = future.result()
                solved_count += 1 if is_solved else 0
                processed_count += 1  
                total_time += elapsed_time 
                print(
                    f"Progress: {processed_count}/{total_problems} processed, "  
                    f"Solved: {solved_count}, "
                    f"Current: {problem_name} "
                    f"({'Success' if is_solved else 'Failed'}), "
                    f"Time: {elapsed_time:.2f}s"
                )

    solved_percentage = (solved_count / total_problems) * 100 if total_problems > 0 else 0
    total_real_time = time.time() - total_real_time
    print(
        f"\nSuccessfully solved {solved_count}/{total_problems} problems ({solved_percentage:.2f}%). "
        f"Total time taken: {total_time:.2f}s, realtime taken: {total_real_time:.2f}s."
    )


def find_free_port(start_port=8000, max_attempts=100):
    """
    Find a free port starting from start_port.
    
    Parameters:
        start_port (int): Starting port number to check
        max_attempts (int): Maximum number of ports to try
        
    Returns:
        int: Available port number
        
    Raises:
        RuntimeError: If no free port is found within max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")

def solve_problem_vllm(args):
    """
    Process a single problem and return whether it was solved successfully along with the time taken.
    """
    problem_name, problems_path, model_path, decoding_size, beam_size, search_depth, api_port = args
    start_time = time.time()
    try:
        solver = (
            GeometricSolverBuilder()
            .load_problem_from_file(problems_path, problem_name)#, rename=True)
            .with_deductive_agent(LMAgent(model_path, decoding_size=decoding_size,beam_size=beam_size, search_depth=search_depth, api_port=api_port))
            .build()
        )
        is_solved = solver.run()
        elapsed_time = time.time() - start_time
        return (problem_name, is_solved, elapsed_time) 
    except Exception as e:
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (problem_name, False, elapsed_time)

def run_newclid_vllm(filepath: Path, modelpath: Path, max_workers: int = 4, decoding_size: int = 1, beam_size: int = 1, search_depth: int = 1):
    """
    Enhanced function that uses VLLM to serve the model before processing problems.
    
    Parameters:
        filepath (Path): The path of the file containing problem names.
        modelpath (Path): Path to the model checkpoint to be served by VLLM.
        max_workers (int): The maximum number of processes in the pool, default is 4.
    """

    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    # Read all problem names (every other line starting from index 0)
    problem_names = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            problem_names.append(lines[i].strip())

    total_problems = len(problem_names)
    print(f"Total problems to solve: {total_problems}")

    # Use ProcessPoolExecutor to process problems concurrently
    solved_count = 0
    processed_count = 0  
    total_time = 0 
    total_real_time = time.time()   

    if max_workers == 1:
        # Single-threaded execution - use the original solve_problem function
        for problem_name in problem_names:
            problem_name, is_solved, elapsed_time = solve_problem((problem_name, filepath, modelpath, decoding_size, beam_size, search_depth))
            solved_count += 1 if is_solved else 0
            processed_count += 1  
            total_time += elapsed_time 
            print(
                f"Progress: {processed_count}/{total_problems} processed, "  
                f"Solved: {solved_count}, "
                f"Current: {problem_name} "
                f"({'Success' if is_solved else 'Failed'}), "
                f"Time: {elapsed_time:.2f}s"
            )
    else:
        # Multi-threaded execution - start VLLM server first
        vllm_process = None
        try:
            available_port = find_free_port()
            print(f"Found available port: {available_port}")
            print(f"Starting VLLM server with model: {modelpath}")
            
            print(f"Starting VLLM server with model: {modelpath}")
            vllm_process = subprocess.Popen([
                "python", "-m", "vllm.entrypoints.api_server",
                "--enable-prefix-caching",
                "--model", str(modelpath),
                "--served-model-name", "Qwen/Qwen2.5-Math-1.5B",
                "--port", str(available_port)
            ])
            
            # 如果要使用CLI命令方式，可以改为：
            # vllm_process = subprocess.Popen([
            #     "vllm", "serve", 
            #     "--model", str(modelpath),
            #     "--port", "8000"
            # ])
            
            # Wait for VLLM server to initialize
            print("Waiting for VLLM server to initialize...")
            time.sleep(10)  # Adjust based on how long your model takes to load
            
            # Multi-threaded execution using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks and collect futures
                futures = {executor.submit(solve_problem, (name, filepath, decoding_size, beam_size, search_depth, available_port)): name for name in problem_names}

                # Process completed tasks
                for future in as_completed(futures):
                    problem_name = futures[future]
                    problem_name, is_solved, elapsed_time = future.result()
                    solved_count += 1 if is_solved else 0
                    processed_count += 1  
                    total_time += elapsed_time 
                    print(
                        f"Progress: {processed_count}/{total_problems} processed, "  
                        f"Solved: {solved_count}, "
                        f"Current: {problem_name} "
                        f"({'Success' if is_solved else 'Failed'}), "
                        f"Time: {elapsed_time:.2f}s"
                    )
                    
        finally:
            # Ensure VLLM server is terminated when we're done
            if vllm_process:
                print("Shutting down VLLM server...")
                vllm_process.send_signal(signal.SIGINT)
                try:
                    vllm_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    vllm_process.kill()
                print("VLLM server shut down")

    solved_percentage = (solved_count / total_problems) * 100 if total_problems > 0 else 0
    total_real_time = time.time() - total_real_time
    print(
        f"\nSuccessfully solved {solved_count}/{total_problems} problems ({solved_percentage:.2f}%). "
        f"Total time taken: {total_time:.2f}s, realtime taken: {total_real_time:.2f}s."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Newclid evaluation with configurable paths.")
    parser.add_argument("--problems_path", type=str, default="problems_datasets/dev_jgex.txt",
                        help="Path to the problems dataset file")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of worker processes to use")
    parser.add_argument("--decoding_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--search_depth", type=int, default=4)
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM to serve the model")
    args = parser.parse_args()
    
    problems_path = Path(args.problems_path)
    model_path = Path(args.model_path)
    
    if args.use_vllm:
        run_newclid_vllm(problems_path, model_path, max_workers=args.max_workers, 
                         decoding_size=args.decoding_size, beam_size=args.beam_size, 
                         search_depth=args.search_depth)
    else:
        run_newclid(problems_path, model_path, max_workers=args.max_workers, 
                    decoding_size=args.decoding_size, beam_size=args.beam_size, 
                    search_depth=args.search_depth)