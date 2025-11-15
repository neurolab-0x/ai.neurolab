from cryptography.fernet import Fernet
import base64
import logging
import time
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Simple in-memory rate limit tracker
rate_limit_tracker = {}

class DataEncryption:
    """Handles data encryption/decryption"""
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt dictionary data"""
        return self.cipher_suite.encrypt(str(data).encode())

    def decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt data back to dictionary"""
        return eval(self.cipher_suite.decrypt(encrypted_data).decode())

class JWTAuth:
    """Simple JWT authentication handler"""
    def __init__(self, secret_key: str = "default_secret_key", token_expiry: int = 24):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.algorithm = "HS256"
    
    def create_token(self, user_id: str, roles: list = None) -> str:
        """Create a JWT token"""
        if roles is None:
            roles = ["user"]
        
        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {str(e)}")

def validate_eeg_data(data) -> bool:
    """Validate EEG data structure"""
    if isinstance(data, np.ndarray):
        return data.size > 0
    if isinstance(data, dict):
        return len(data) > 0
    return False

def validate_client_id(client_id: str) -> bool:
    """Validate client ID format"""
    if not client_id or len(client_id) > 64:
        return False
    return all(c.isalnum() or c in '_-.' for c in client_id)

def check_rate_limit(client_id: str, endpoint: str, max_requests: int, window_seconds: int) -> bool:
    """Check if client has exceeded rate limit"""
    key = f"{client_id}:{endpoint}"
    current_time = time.time()
    
    if key not in rate_limit_tracker:
        rate_limit_tracker[key] = []
    
    # Remove old requests outside the window
    rate_limit_tracker[key] = [
        req_time for req_time in rate_limit_tracker[key]
        if current_time - req_time < window_seconds
    ]
    
    # Check if limit exceeded
    if len(rate_limit_tracker[key]) >= max_requests:
        return False
    
    # Add current request
    rate_limit_tracker[key].append(current_time)
    return True

def sanitize_model_type(model_type: str) -> str:
    """Sanitize model type input"""
    if not model_type:
        return "original"
    return model_type.lower().strip() 