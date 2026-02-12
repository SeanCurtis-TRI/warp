import re
import subprocess

def set_required_parameters(param_dict):
    """
    Ensures that all required parameters are set in the dictionary, possibly
    overriding any existing values.

    Args:
        param_dict (dict): The dictionary of parameters to check.
    """
    param_dict["headless"] = None
    param_dict["num_frames"] = 1000
    param_dict["scenario"] = "circle"
    param_dict["domain_size"] = 130
    param_dict["run_all_frames"] = None

def param_dict_to_list(param_dict):
    """
    Converts a dictionary of parameters to a list of command-line arguments.

    Args:
        param_dict (dict): A dictionary where keys are parameter names and values are parameter values.
                           If a value is None, it will be treated as a flag (no value).

    Returns:
        list: A list of command-line arguments.
    """
    args = []
    for key, value in param_dict.items():
        if value is None:
            args.append(f"--{key}")  # Flag without a value
        else:
            args.append(f"--{key}")
            args.append(str(value))  # Convert the value to string
    return args

def iteration_header(param_dict):
    """
    Creates a string calling out the varying parameters.
    """
    default_params = {}
    set_required_parameters(default_params)

    varying_params = {k: v for k, v in param_dict.items() if k not in default_params}
    return ", ".join(f"{k}, {v}" for k, v in varying_params.items())


def extract_output(stdout_str):
    """
    Extracts interesting information from the output string.

    Args:
        stdout_str (str): The captured standard output from the script.

    Returns:
        str: A formatted string containing the relevant information, or an
             error message if not found.
    """
    lines = stdout_str.splitlines()
    data = {}
    population_re = re.compile(r"Populating with (\d+) agents")
    timing_re = re.compile(r"Total time for (\d+) frames: ([\d.]+) seconds")
    for line in lines:
        m = population_re.match(line)
        if m:
            data["population"] = m.group(1)
        else:
            m = timing_re.match(line)
            if m:
                data["frames"] = m.group(1)
                data["time"] = m.group(2)
    if "population" not in data:
        data["population"]  = "No population information found"
    if "time" not in data:
        data["time"] = "No timing information found"
    return (f"Actual population, {data['population']}, "
            f"Actual frames, {data['frames']}, "
            f"Time, {data['time']}")

def run_script_repeatedly(script_path, parameters_list):
    """
    Runs another script with different parameters and captures stdout.

    Args:
        script_path (str): The path to the script you want to run.
        parameters_list (list of dictionaries): Each dictionary is a set of
                                               key-value pairs representing a
                                               parameter. None is considered a
                                               valid value, which means no
                                               value.
    """
    for params in parameters_list:
        set_required_parameters(params)
        # Build the full command
        command = ['python', script_path] + param_dict_to_list(params)
        
        try:
            # Run the command and capture the output
            result = subprocess.run(
                command,
                check=True,                 # Raise an exception if the script fails (non-zero exit code)
                stdout=subprocess.PIPE,     # Capture stdout
                stderr=subprocess.PIPE,     # Capture stderr (optional, good practice)
                text=True                   # Decode output as text (str), not bytes
            )
            print(f"{iteration_header(params)}, {extract_output(result.stdout)}")
                
        except subprocess.CalledProcessError as e:
            print(f"Script failed with exit code {e.returncode}")
            print("STDOUT:")
            print(e.stdout)
            print("STDERR:")
            print(e.stderr)
        except FileNotFoundError:
            print(f"Error: The script or Python interpreter was not found.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

# --- Example Usage ---
if __name__ == "__main__":
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "sean_crowd.py"
    all_parameters = []

    for use_density in [True, False]:
        params = {}
        if not use_density:
            params["no_density"] = None
        for agent_count in [250, 500, 1000, 2000, 4000, 8000]:
            params["num_agents"] = agent_count
            for grid_size in [0, 16, 24, 32, 48, 64]:
                params["grid_size"] = grid_size
                all_parameters.append(params.copy())
    
    # 3. Run the function
    run_script_repeatedly(script_path, all_parameters)
