#!/usr/bin/env python3
"""
Debugging and Monitoring System for Spaced Repetition
This module provides comprehensive logging, debugging, and monitoring capabilities.
"""

import logging
import json
import traceback
from datetime import datetime, timezone
from functools import wraps
import sys
import os

# Configure logging
def setup_debugging():
    """Set up comprehensive logging system"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/spaced_repetition.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create specific loggers
    spaced_repetition_logger = logging.getLogger('spaced_repetition')
    spaced_repetition_logger.setLevel(logging.DEBUG)
    
    api_logger = logging.getLogger('spaced_repetition.api')
    api_logger.setLevel(logging.INFO)
    
    algorithm_logger = logging.getLogger('spaced_repetition.algorithm')
    algorithm_logger.setLevel(logging.DEBUG)
    
    session_logger = logging.getLogger('spaced_repetition.session')
    session_logger.setLevel(logging.INFO)
    
    return {
        'main': spaced_repetition_logger,
        'api': api_logger,
        'algorithm': algorithm_logger,
        'session': session_logger
    }

# Initialize loggers
loggers = setup_debugging()

def debug_logger(func):
    """Decorator to add comprehensive debugging to functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = loggers['main']
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args: {args}, kwargs: {kwargs}")
        
        try:
            start_time = datetime.now(timezone.utc)
            result = func(*args, **kwargs)
            end_time = datetime.now(timezone.utc)
            
            # Log successful completion
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"Completed {func.__name__} in {duration:.3f}s with result: {result}")
            
            return result
            
        except Exception as e:
            # Log error with full traceback
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create error report
            error_report = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Save error report to file
            with open(f'logs/error_{datetime.now().strftime("%Y%m%d")}.json', 'a') as f:
                f.write(json.dumps(error_report) + '\n')
            
            raise
    
    return wrapper

class SpacedRepetitionDebugger:
    """Centralized debugging and monitoring class"""
    
    def __init__(self):
        self.logger = loggers['main']
        self.api_logger = loggers['api']
        self.algorithm_logger = loggers['algorithm']
        self.session_logger = loggers['session']
        
    def log_api_call(self, endpoint, method, user_id, data=None, response=None):
        """Log API calls with full context"""
        self.api_logger.info(f"API Call: {method} {endpoint}")
        self.api_logger.info(f"User: {user_id}")
        if data:
            self.api_logger.debug(f"Request Data: {json.dumps(data, default=str)}")
        if response:
            self.api_logger.debug(f"Response: {json.dumps(response, default=str)}")
    
    def log_algorithm_calculation(self, card_id, old_interval, quality_rating, new_interval, ease_factor):
        """Log spaced repetition algorithm calculations"""
        self.algorithm_logger.info(f"Card {card_id}: {old_interval}d -> {new_interval}d (rating: {quality_rating}, ease: {ease_factor:.2f})")
    
    def log_session_event(self, session_id, event_type, data=None):
        """Log session events"""
        self.session_logger.info(f"Session {session_id}: {event_type}")
        if data:
            self.session_logger.debug(f"Event Data: {json.dumps(data, default=str)}")
    
    def log_performance_metric(self, metric_name, value, context=None):
        """Log performance metrics"""
        self.logger.info(f"Performance: {metric_name} = {value}")
        if context:
            self.logger.debug(f"Context: {json.dumps(context, default=str)}")
    
    def create_debug_report(self):
        """Create comprehensive debug report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            },
            'log_files': self._get_log_file_info(),
            'recent_errors': self._get_recent_errors()
        }
        
        # Save debug report
        with open(f'logs/debug_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _get_log_file_info(self):
        """Get information about log files"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            return []
        
        files = []
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                files.append({
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return files
    
    def _get_recent_errors(self):
        """Get recent error information"""
        error_file = f'logs/error_{datetime.now().strftime("%Y%m%d")}.json'
        if not os.path.exists(error_file):
            return []
        
        errors = []
        try:
            with open(error_file, 'r') as f:
                for line in f:
                    if line.strip():
                        errors.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"Error reading error file: {e}")
        
        return errors[-10:]  # Return last 10 errors

# Global debugger instance
debugger = SpacedRepetitionDebugger()

def log_user_action(user_id, action, details=None):
    """Log user actions for debugging"""
    debugger.logger.info(f"User {user_id}: {action}")
    if details:
        debugger.logger.debug(f"Details: {json.dumps(details, default=str)}")

def log_system_health():
    """Log system health metrics"""
    import psutil
    
    health_metrics = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    debugger.log_performance_metric('system_health', health_metrics)
    return health_metrics

if __name__ == "__main__":
    # Test the debugging system
    print("Testing Spaced Repetition Debugging System...")
    
    # Test logging
    debugger.log_api_call('/api/test', 'POST', 'test_user', {'test': 'data'})
    debugger.log_algorithm_calculation('card_123', 1, 2, 6, 2.5)
    debugger.log_session_event('session_456', 'started', {'cards': 20})
    
    # Test error handling
    try:
        raise Exception("Test error for debugging")
    except Exception as e:
        debugger.logger.error(f"Test error caught: {e}")
    
    # Create debug report
    report = debugger.create_debug_report()
    print(f"Debug report created: {report['timestamp']}")
    
    print("Debugging system test complete!")
