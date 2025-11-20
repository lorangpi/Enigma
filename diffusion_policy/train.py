"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
import wandb
import psutil
import time
import threading
import subprocess
import logging
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# --------------------------------------------------------------------------------------
# System monitoring functions
# --------------------------------------------------------------------------------------
def get_gpu_power_usage():
    """Get GPU power usage in watts."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            power_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(power_values) if power_values else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_utilization():
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            util_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(util_values) / len(util_values) if util_values else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_memory_usage():
    """Get GPU memory usage percentage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            total_used = 0
            total_available = 0
            for line in lines:
                if ',' in line:
                    used, total = line.split(',')
                    total_used += float(used.strip())
                    total_available += float(total.strip())
            return (total_used / total_available * 100) if total_available > 0 else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_cpu_power_usage():
    """Get CPU power usage estimation based on frequency and utilization."""
    try:
        # Get CPU frequency and utilization
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if cpu_freq and cpu_freq.current > 0:
            # Rough estimation: higher frequency and utilization = higher power
            # This is a simplified model - actual power depends on many factors
            base_power = 15.0  # Base power in watts
            freq_factor = (cpu_freq.current / cpu_freq.max) if cpu_freq.max > 0 else 1.0
            util_factor = cpu_percent / 100.0
            
            estimated_power = base_power * freq_factor * (0.5 + 0.5 * util_factor)
            return estimated_power
    except Exception:
        pass
    return 0.0

def get_system_metrics():
    """Get comprehensive system metrics with timestamps."""
    try:
        # Get current timestamp
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # GPU metrics
        gpu_power = get_gpu_power_usage()
        gpu_util = get_gpu_utilization()
        gpu_memory = get_gpu_memory_usage()
        
        # CPU power estimation
        cpu_power = get_cpu_power_usage()
        
        return {
            'timestamp': current_time,
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
            'cpu_percent': cpu_percent,
            'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
            'cpu_count': cpu_count,
            'cpu_power_watts': cpu_power,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'disk_percent': disk_percent,
            'disk_used_gb': disk_used_gb,
            'disk_total_gb': disk_total_gb,
            'gpu_power_watts': gpu_power,
            'gpu_utilization_percent': gpu_util,
            'gpu_memory_percent': gpu_memory,
        }
    except Exception as e:
        logging.warning(f"Error getting system metrics: {e}")
        return {}

def calculate_energy_consumption(power_data_points):
    """
    Calculate total energy consumption from power data points with timestamps.
    
    Args:
        power_data_points: List of dicts with 'timestamp' and power values
        
    Returns:
        Dict with total energy consumption in watt-hours and joules
    """
    if len(power_data_points) < 2:
        return {'total_energy_wh': 0, 'total_energy_joules': 0, 'duration_seconds': 0}
    
    # Sort by timestamp
    sorted_points = sorted(power_data_points, key=lambda x: x['timestamp'])
    
    total_gpu_energy = 0
    total_cpu_energy = 0
    total_duration = 0
    
    for i in range(1, len(sorted_points)):
        prev = sorted_points[i-1]
        curr = sorted_points[i]
        
        # Time interval in hours
        time_interval_hours = (curr['timestamp'] - prev['timestamp']) / 3600.0
        time_interval_seconds = curr['timestamp'] - prev['timestamp']
        
        # Average power during this interval
        avg_gpu_power = (prev.get('gpu_power_watts', 0) + curr.get('gpu_power_watts', 0)) / 2
        avg_cpu_power = (prev.get('cpu_power_watts', 0) + curr.get('cpu_power_watts', 0)) / 2
        
        # Energy = Power × Time
        gpu_energy = avg_gpu_power * time_interval_hours
        cpu_energy = avg_cpu_power * time_interval_hours
        
        total_gpu_energy += gpu_energy
        total_cpu_energy += cpu_energy
        total_duration += time_interval_seconds
    
    total_energy_wh = total_gpu_energy + total_cpu_energy
    total_energy_joules = total_energy_wh * 3600  # Convert watt-hours to joules
    
    return {
        'total_energy_wh': total_energy_wh,
        'total_energy_joules': total_energy_joules,
        'gpu_energy_wh': total_gpu_energy,
        'cpu_energy_wh': total_cpu_energy,
        'duration_seconds': total_duration,
        'duration_hours': total_duration / 3600
    }

class SystemMonitor:
    """Comprehensive system monitoring with GPU, CPU, memory, and energy tracking"""
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.power_data_points = []
        
    def start_monitoring(self):
        """Start system monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring and calculate final energy consumption"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Calculate final energy consumption
        if len(self.power_data_points) > 1:
            energy_stats = calculate_energy_consumption(self.power_data_points)
            print(f"\n=== Energy Consumption Summary ===")
            print(f"Total Energy: {energy_stats['total_energy_wh']:.2f} Wh ({energy_stats['total_energy_joules']:.0f} J)")
            print(f"GPU Energy: {energy_stats['gpu_energy_wh']:.2f} Wh")
            print(f"CPU Energy: {energy_stats['cpu_energy_wh']:.2f} Wh")
            print(f"Duration: {energy_stats['duration_hours']:.2f} hours")
            
            # Log final energy stats to wandb
            wandb.log({
                "final_total_energy_wh": energy_stats['total_energy_wh'],
                "final_total_energy_joules": energy_stats['total_energy_joules'],
                "final_gpu_energy_wh": energy_stats['gpu_energy_wh'],
                "final_cpu_energy_wh": energy_stats['cpu_energy_wh'],
                "final_duration_hours": energy_stats['duration_hours']
            })
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get comprehensive system metrics
                metrics = get_system_metrics()
                
                if metrics:  # Only proceed if metrics were successfully collected
                    # Store power data for energy calculation
                    self.power_data_points.append(metrics.copy())
                    
                    # Log to wandb
                    wandb.log(metrics)
                    
                    # Print current status
                    print(f"[{metrics['timestamp_iso']}] CPU: {metrics['cpu_percent']:.1f}% "
                          f"({metrics['cpu_freq_mhz']:.0f}MHz), Memory: {metrics['memory_percent']:.1f}%, "
                          f"GPU: {metrics['gpu_utilization_percent']:.1f}% "
                          f"({metrics['gpu_power_watts']:.1f}W), GPU Mem: {metrics['gpu_memory_percent']:.1f}%")
                
                time.sleep(self.log_interval)
            except Exception as e:
                print(f"Error in system monitoring: {e}")
                time.sleep(self.log_interval)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # Initialize wandb
    wandb.init(
        project="FINAL_diffusion-policy-training",
        name=f"diffusion_policy_{int(time.time())}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["diffusion_policy", "system_monitoring", "energy_tracking"]
    )
    
    # Initialize system monitor
    system_monitor = SystemMonitor(log_interval=10)  # Log every 10 seconds
    system_monitor.start_monitoring()
    
    try:
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Stop system monitoring and finish wandb
        system_monitor.stop_monitoring()
        wandb.finish()

if __name__ == "__main__":
    main()
