import pandas as pd
import os
from datetime import datetime

DB_FILE = "database.xlsx"
IMG_DIR = "images/gallery"


# =========================
# INIT
# =========================
def init_database():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs("images/breach_logs", exist_ok=True)

    if not os.path.exists(DB_FILE):
        with pd.ExcelWriter(DB_FILE, engine="openpyxl") as writer:
            # Admin sheet
            pd.DataFrame(columns=[
                "user_id",
                "name",
                "email",
                "age",
                "gender",
                "phone",
                "dept",
                "face_encoding",
                "user_type",
                "admin_pin",
                "created_at",
                "last_login",
                "last_logout"
            ]).to_excel(writer, sheet_name="admins", index=False)
            
            # Users sheet
            pd.DataFrame(columns=[
                "user_id",
                "name",
                "email",
                "age",
                "gender",
                "phone",
                "dept",
                "face_encoding",
                "user_type",
                "admin_pin",
                "created_at",
                "last_login",
                "last_logout"
            ]).to_excel(writer, sheet_name="users", index=False)


# =========================
# DB HELPERS
# =========================
def load_db():
    """Load both admin and user sheets and combine them"""
    try:
        admins = pd.read_excel(DB_FILE, sheet_name="admins")
        users = pd.read_excel(DB_FILE, sheet_name="users")
        return pd.concat([admins, users], ignore_index=True)
    except:
        init_database()
        return load_db()


def save_db(df):
    """Save dataframe back to appropriate sheets"""
    admins = df[df["user_type"] == "admin"]
    users = df[df["user_type"] == "general_user"]
    
    with pd.ExcelWriter(DB_FILE, engine="openpyxl", mode="w") as writer:
        admins.to_excel(writer, sheet_name="admins", index=False)
        users.to_excel(writer, sheet_name="users", index=False)


# =========================
# USER CRUD
# =========================
def register_user(user_id, name, email, age, gender, phone, dept, encoding, user_type="general_user", admin_pin=None):
    """Register a new user with predefined user_id"""
    df = load_db()

    df = pd.concat([df, pd.DataFrame([{
        "user_id": user_id,
        "name": name,
        "email": email,
        "age": age,
        "gender": gender,
        "phone": phone,
        "dept": dept,
        "face_encoding": encoding,
        "user_type": user_type,
        "admin_pin": admin_pin if user_type == "admin" else None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "Never",
        "last_logout": "Never"
    }])], ignore_index=True)

    save_db(df)
    return user_id


def get_all_users():
    """Get all users (both admins and general users)"""
    return load_db()


def get_users_by_type(user_type):
    """Get users by type (admin or general_user)"""
    df = load_db()
    return df[df["user_type"] == user_type]


def update_user_details(uid, name, email, age, phone, dept):
    """Update all user details at once"""
    df = load_db()
    idx = df.index[df["user_id"] == uid]
    if idx.empty:
        return False

    df.loc[idx[0], ["name", "email", "age", "phone", "dept"]] = [name, email, age, phone, dept]
    save_db(df)
    return True


def update_user_field(uid, field, value):
    """Update a single user field"""
    df = load_db()
    idx = df.index[df["user_id"] == uid]
    if idx.empty:
        return False

    df.loc[idx[0], field] = value
    save_db(df)
    return True


def update_admin_pin(uid, new_pin):
    """Update admin PIN"""
    df = load_db()
    idx = df.index[df["user_id"] == uid]
    if idx.empty:
        return False

    df.loc[idx[0], "admin_pin"] = new_pin
    save_db(df)
    return True


def update_login_timestamp(uid):
    """Update last login timestamp"""
    df = load_db()
    idx = df.index[df["user_id"] == uid]
    if idx.empty:
        return False

    df.loc[idx[0], "last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_db(df)
    return True


def update_logout_timestamp(uid):
    """Update last logout timestamp"""
    df = load_db()
    idx = df.index[df["user_id"] == uid]
    if idx.empty:
        return False

    df.loc[idx[0], "last_logout"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_db(df)
    return True


def delete_user(uid):
    """Delete user and their images"""
    df = load_db()
    df = df[df["user_id"] != uid]
    save_db(df)

    user_dir = os.path.join(IMG_DIR, uid)
    if os.path.exists(user_dir):
        import shutil
        shutil.rmtree(user_dir)
    return True


# =========================
# IMAGE CRUD
# =========================
def get_user_images(uid):
    """Get all images for a user"""
    user_dir = os.path.join(IMG_DIR, uid)
    if not os.path.exists(user_dir):
        return []
    return [os.path.join(user_dir, f) for f in sorted(os.listdir(user_dir))]


def add_user_image(uid, image_path):
    """Add image to user gallery"""
    user_dir = os.path.join(IMG_DIR, uid)
    os.makedirs(user_dir, exist_ok=True)
    return True


def delete_user_image(image_path):
    """Delete a user image"""
    if os.path.exists(image_path):
        os.remove(image_path)
        return True
    return False
