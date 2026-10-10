import bcrypt
from core.db_setup import get_db

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    # Convert password to bytes
    password_bytes = password.encode('utf-8')
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a hashed password."""
    plain_password_bytes = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_password_bytes, hashed_password_bytes)

def register_user(username, password):
    """Registers a new user in the database."""
    db = get_db()
    users_collection = db['users']
    
    import re
    # Check if user already exists (case-insensitive check to be more robust)
    if users_collection.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}):
        return False, "Username already exists."
    
    # Hash password and insert
    hashed_password = hash_password(password)
    new_user = {
        "username": username,
        "password": hashed_password
    }
    
    try:
        users_collection.insert_one(new_user)
        return True, "User registered successfully."
    except Exception as e:
        return False, f"An error occurred: {e}"

def authenticate_user(username, password):
    """Authenticates a user by checking credentials against the database."""
    db = get_db()
    users_collection = db['users']
    
    user = users_collection.find_one({"username": username})
    
    if user:
        if verify_password(password, user['password']):
            return True, "Authentication successful."
    
    return False, "Invalid username or password."
