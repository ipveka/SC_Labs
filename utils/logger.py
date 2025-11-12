"""
Logging utility for SC Labs
"""

import sys
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Logger:
    """Simple logger with clean formatting"""
    
    def __init__(self, module_name, use_colors=True, show_timestamps=False):
        self.module_name = module_name
        self.use_colors = use_colors
        self.show_timestamps = show_timestamps
    
    def _format_message(self, level, message, color=''):
        """Format log message"""
        timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] " if self.show_timestamps else ""
        
        if self.use_colors and color:
            return f"{timestamp}{color}[{self.module_name}] {level}: {message}{Colors.END}"
        else:
            return f"{timestamp}[{self.module_name}] {level}: {message}"
    
    def info(self, message):
        """Log info message"""
        print(self._format_message("INFO", message, Colors.BLUE))
    
    def success(self, message):
        """Log success message"""
        print(self._format_message("✓", message, Colors.GREEN))
    
    def warning(self, message):
        """Log warning message"""
        print(self._format_message("WARNING", message, Colors.YELLOW))
    
    def error(self, message):
        """Log error message"""
        print(self._format_message("ERROR", message, Colors.RED), file=sys.stderr)
    
    def step(self, step_num, total_steps, message):
        """Log step in a process"""
        if self.use_colors:
            print(f"{Colors.CYAN}[{step_num}/{total_steps}]{Colors.END} {message}")
        else:
            print(f"[{step_num}/{total_steps}] {message}")
    
    def metric(self, name, value, unit=''):
        """Log a metric"""
        unit_str = f" {unit}" if unit else ""
        if self.use_colors:
            print(f"  {Colors.BOLD}{name}:{Colors.END} {value}{unit_str}")
        else:
            print(f"  {name}: {value}{unit_str}")
    
    def section(self, title):
        """Log section header"""
        separator = "=" * 60
        if self.use_colors:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{separator}")
            print(f"{title}")
            print(f"{separator}{Colors.END}\n")
        else:
            print(f"\n{separator}")
            print(f"{title}")
            print(f"{separator}\n")


def suppress_warnings():
    """Suppress common warnings"""
    import warnings
    import logging
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    
    # Suppress GluonTS/MXNet logging
    logging.getLogger('gluonts').setLevel(logging.ERROR)
    logging.getLogger('mxnet').setLevel(logging.ERROR)
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('lightning').setLevel(logging.ERROR)
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
