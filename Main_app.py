import customtkinter as ctk
import tkinter.messagebox as msg
from tkinter import filedialog
import cv2
import json
import os
import re
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import face_recognition
from datetime import datetime, timedelta
import shutil
import uuid

import database_manager as db
from face_system import FaceSystem

ctk.set_appearance_mode("Dark")


class FaceApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Fetch Details Of Person Based on Face Values")
        self.geometry("1600x900")
        self.configure(fg_color="#0a0a0a")
        
        # Initialize variables first
        self.bg_label = None
        self.bg_photo = None
        self.cap = None
        self.current_frame = None
        self.current_user = None
        self.current_user_type = None
        self.last_face_status = "Initializing"
        self.face_locations = []
        self.login_attempts = {}
        self.image_captured = False
        self.captured_encoding = None
        self.captured_frame = None
        
        # Set background
        self.set_background()

        # Initialize system
        db.init_database()
        self.init_folder_structure()
        self.cleanup_old_data()
        
        self.face = FaceSystem()

        self.build_home()

    def init_folder_structure(self):
        """Initialize complete folder structure"""
        folders = [
            "images",
            "images/gallery",
            "images/breach_logs"
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def cleanup_old_data(self):
        """Auto-cleanup old login images and breach logs"""
        try:
            breach_base = "images/breach_logs"
            if os.path.exists(breach_base):
                cutoff_date = datetime.now() - timedelta(days=30)
                for date_folder in os.listdir(breach_base):
                    folder_path = os.path.join(breach_base, date_folder)
                    if os.path.isdir(folder_path):
                        try:
                            folder_date = datetime.strptime(date_folder, "%Y-%m-%d")
                            if folder_date < cutoff_date:
                                shutil.rmtree(folder_path)
                        except:
                            pass
            
            gallery_base = "images/gallery"
            if os.path.exists(gallery_base):
                cutoff_date = datetime.now() - timedelta(days=30)
                for user_folder in os.listdir(gallery_base):
                    user_path = os.path.join(gallery_base, user_folder)
                    if os.path.isdir(user_path):
                        for img_file in os.listdir(user_path):
                            if img_file.startswith("login_"):
                                img_path = os.path.join(user_path, img_file)
                                file_time = datetime.fromtimestamp(os.path.getmtime(img_path))
                                if file_time < cutoff_date:
                                    os.remove(img_path)
        except Exception as e:
            print(f"Cleanup error: {e}")

    def set_background(self):
        """Set background image with optimization"""
        bg_files = ["background.jpeg", "background.jpg", "background.png"]
        for bg_file in bg_files:
            if os.path.exists(bg_file):
                try:
                    bg_img = Image.open(bg_file)
                    if bg_img.size[0] > 1920 or bg_img.size[1] > 1080:
                        bg_img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                    bg_img = bg_img.resize((1600, 900), Image.Resampling.LANCZOS)
                    self.bg_photo = ImageTk.PhotoImage(bg_img)
                    if self.bg_label is None:
                        self.bg_label = ctk.CTkLabel(self, image=self.bg_photo, text="")
                    else:
                        self.bg_label.configure(image=self.bg_photo)
                    self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                    return
                except Exception as e:
                    print(f"Background error: {e}")
        
        # Create gradient background if no image found
        self.configure(fg_color="#0a0a0a")

    def clear_screen(self):
        """Clear screen but keep background"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.image_captured = False
        
        for w in self.winfo_children():
            if w != self.bg_label:
                w.destroy()

    def start_camera(self):
        """Start camera capture"""
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                msg.showerror("Error", "Could not open camera")
                return
        self.update_camera()

    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_camera(self):
        """Update camera feed"""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self.update_camera)
            return

        self.current_frame = frame.copy()
        enc, status, face_locs = self.face.process(frame)
        self.last_face_status = status
        self.face_locations = face_locs
        
        display_frame = frame.copy()
        if face_locs:
            for (top, right, bottom, left) in face_locs:
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                color = (0, 255, 0) if status == "Face OK" else (0, 165, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)
        
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ctk.CTkImage(img, size=(640, 480))
        
        if hasattr(self, 'camera_label') and self.camera_label.winfo_exists():
            self.camera_label.configure(image=imgtk, text="")
            self.camera_label.image = imgtk

        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            self.status_label.configure(text=status)

        if hasattr(self, "btn_capture") and self.btn_capture.winfo_exists():
            self.btn_capture.configure(state="normal" if status == "Face OK" else "disabled")

        self.after(30, self.update_camera)

    # ================= VALIDATION =================
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_phone(self, phone):
        """Validate phone number (10 digits)"""
        pattern = r'^\d{10}$'
        return re.match(pattern, phone) is not None

    def validate_name(self, name):
        """Validate full name (letters and spaces only, min 2 words)"""
        pattern = r'^[a-zA-Z]+(?: [a-zA-Z]+)+$'
        return re.match(pattern, name.strip()) is not None

    def validate_age(self, age):
        """Validate age (1-120)"""
        try:
            age_int = int(age)
            return 1 <= age_int <= 120
        except:
            return False

    def validate_department(self, dept):
        """Validate department (not empty, min 2 characters)"""
        return len(dept.strip()) >= 2

    def validate_registration_form(self, user_type):
        """Validate all registration fields"""
        errors = []
        
        # Name validation
        name = self.entry_name.get().strip()
        if not name:
            errors.append("Please fill 'Full Name'")
        elif not self.validate_name(name):
            errors.append("Full Name must contain at least first and last name (letters only)")
        
        # Email validation
        email = self.entry_email.get().strip()
        if not email:
            errors.append("Please fill 'Email ID'")
        elif not self.validate_email(email):
            errors.append("Email ID must be in proper format (e.g., user@example.com)")
        
        # Age validation
        age = self.entry_age.get().strip()
        if not age:
            errors.append("Please fill 'Age'")
        elif not self.validate_age(age):
            errors.append("Age must be a number between 1 and 120")
        
        # Gender validation
        gender = self.gender_var.get()
        if not gender or gender == "Select Gender":
            errors.append("Please select 'Gender'")
        
        # Phone validation
        phone = self.entry_phone.get().strip()
        if not phone:
            errors.append("Please fill 'Phone Number'")
        elif not self.validate_phone(phone):
            errors.append("Phone Number must be exactly 10 digits")
        
        # Department validation
        dept = self.entry_dept.get().strip()
        if not dept:
            errors.append("Please fill 'Department'")
        elif not self.validate_department(dept):
            errors.append("Department must be at least 2 characters")
        
        # Admin PIN validation
        if user_type == "admin":
            pin = self.entry_pin.get().strip()
            if not pin:
                errors.append("Please fill 'Admin PIN'")
            elif len(pin) != 4 or not pin.isdigit():
                errors.append("Admin PIN must be exactly 4 digits")
        
        return errors

    # ================= HOME =================
    def build_home(self):
        """Build home screen"""
        self.clear_screen()
        self.set_background()

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.pack(fill="x", padx=30, pady=20)

        ctk.CTkLabel(
            header, 
            text="IDENTITY FETCH .", 
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff",
            anchor="w"
        ).pack(side="left")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")
        
        ctk.CTkButton(
            btn_frame,
            text="Identify",
            command=self.show_login,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Register",
            command=self.show_register_choice,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)

        # Center content
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            center_frame,
            text="Verify Identity",
            font=("Segoe UI", 48, "bold"),
            text_color="#ffffff"
        ).pack(pady=(0, 50))

        ctk.CTkButton(
            center_frame,
            text="Start Face Scan",
            command=self.show_login,
            fg_color="#3b7cb8",
            hover_color="#2d5f8d",
            text_color="#ffffff",
            width=450,
            height=60,
            corner_radius=50,
            font=("Segoe UI", 18, "bold")
        ).pack(pady=15)

        ctk.CTkButton(
            center_frame,
            text="Manager Login",
            command=self.show_login,
            fg_color="#6b2d8c",
            hover_color="#532270",
            text_color="#ffffff",
            width=450,
            height=60,
            corner_radius=50,
            font=("Segoe UI", 18, "bold")
        ).pack(pady=15)

    # ================= REGISTER CHOICE =================
    def show_register_choice(self):
        """Show registration type selection"""
        self.clear_screen()
        self.set_background()

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.pack(fill="x", padx=30, pady=20)

        ctk.CTkLabel(
            header,
            text="IDENTITY FETCH .",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff",
            anchor="w"
        ).pack(side="left")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")
        
        ctk.CTkButton(
            btn_frame,
            text="Identify",
            command=self.show_login,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Register",
            command=self.show_register_choice,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)

        # Center card
        card = ctk.CTkFrame(
            self,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=25,
            width=600,
            height=500
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)

        ctk.CTkLabel(
            card,
            text="Choose Registration Type",
            font=("Segoe UI", 32, "bold"),
            text_color="#ffffff"
        ).pack(pady=(50, 10))

        # Separator
        separator = ctk.CTkFrame(card, height=2, fg_color="#444444")
        separator.pack(fill="x", padx=100, pady=10)

        ctk.CTkButton(
            card,
            text="Register as Admin",
            command=lambda: self.show_register("admin"),
            fg_color="#2d1f3d",
            hover_color="#3d2f4d",
            border_width=2,
            border_color="#6b2d8c",
            text_color="#ffffff",
            width=400,
            height=55,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(pady=20)

        ctk.CTkButton(
            card,
            text="Register as General User",
            command=lambda: self.show_register("general_user"),
            fg_color="#1f2d3d",
            hover_color="#2f3d4d",
            border_width=2,
            border_color="#3b7cb8",
            text_color="#ffffff",
            width=400,
            height=55,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(pady=20)

        ctk.CTkButton(
            card,
            text="Back",
            command=self.build_home,
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            border_width=2,
            border_color="#555555",
            text_color="#ffffff",
            width=250,
            height=45,
            corner_radius=50,
            font=("Segoe UI", 14)
        ).pack(pady=30)

    # ================= REGISTER =================
    def show_register(self, user_type):
        """Show registration form"""
        self.clear_screen()
        self.set_background()
        self.image_captured = False

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.pack(fill="x", padx=30, pady=20)

        ctk.CTkLabel(
            header,
            text="IDENTITY FETCH .",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff",
            anchor="w"
        ).pack(side="left")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")
        
        ctk.CTkButton(
            btn_frame,
            text="Identify",
            command=self.show_login,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Register",
            command=self.show_register_choice,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)

        # Main card with side-by-side layout
        card = ctk.CTkFrame(
            self,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=25
        )
        card.place(relx=0.5, rely=0.55, anchor="center")

        title = "Admin Registration" if user_type == "admin" else "User Registration"
        ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff"
        ).pack(pady=(20, 10))

        # Separator
        separator = ctk.CTkFrame(card, height=2, fg_color="#444444")
        separator.pack(fill="x", padx=100, pady=5)

        # Side-by-side container
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # LEFT SIDE - Form (Scrollable)
        left_frame = ctk.CTkScrollableFrame(
            content_frame, 
            fg_color="transparent",
            width=420,
            height=500
        )
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        new_user_id = f"{'ADM' if user_type == 'admin' else 'USR'}-{uuid.uuid4().hex[:6].upper()}"
        
        # User ID (Auto-generated)
        self.entry_userid = ctk.CTkEntry(
            left_frame,
            placeholder_text="User ID (Auto-generated)",
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            text_color="#888888",
            font=("Segoe UI", 12)
        )
        self.entry_userid.insert(0, new_user_id)
        self.entry_userid.configure(state="disabled")
        self.entry_userid.pack(pady=6)

        # Full Name
        self.entry_name = ctk.CTkEntry(
            left_frame,
            placeholder_text="Full Name *",
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_name.pack(pady=6)

        # Email
        self.entry_email = ctk.CTkEntry(
            left_frame,
            placeholder_text="Email ID *",
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_email.pack(pady=6)

        # Age
        self.entry_age = ctk.CTkEntry(
            left_frame,
            placeholder_text="Age *",
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_age.pack(pady=6)

        # Gender Dropdown
        self.gender_var = ctk.StringVar(value="Select Gender")
        self.gender_dropdown = ctk.CTkOptionMenu(
            left_frame,
            variable=self.gender_var,
            values=["Male", "Female"],
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            button_color="#3b7cb8",
            button_hover_color="#2d5f8d",
            dropdown_fg_color="#2a2a2a",
            dropdown_hover_color="#3a3a3a",
            font=("Segoe UI", 12)
        )
        self.gender_dropdown.pack(pady=6)

        # Phone Number
        self.entry_phone = ctk.CTkEntry(
            left_frame,
            placeholder_text="Phone Number (10 digits) *",
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_phone.pack(pady=6)

        # Department
        self.entry_dept = ctk.CTkEntry(
            left_frame,
            placeholder_text="Department *",
            width=380,
            height=42,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_dept.pack(pady=6)

        # Admin PIN (only for admin)
        if user_type == "admin":
            self.entry_pin = ctk.CTkEntry(
                left_frame,
                placeholder_text="Admin PIN (4 digits) *",
                width=380,
                height=42,
                corner_radius=50,
                fg_color="#2a2a2a",
                border_width=2,
                border_color="#333333",
                show="*",
                font=("Segoe UI", 12)
            )
            self.entry_pin.pack(pady=6)

        # Required fields note
        ctk.CTkLabel(
            left_frame,
            text="* Required fields",
            font=("Segoe UI", 10),
            text_color="#888888"
        ).pack(pady=5)

        # Capture Button
        self.btn_capture = ctk.CTkButton(
            left_frame,
            text="📷 Capture from Webcam",
            command=lambda: self.start_camera_capture(user_type, new_user_id),
            fg_color="#1f2d3d",
            hover_color="#2f3d4d",
            border_width=2,
            border_color="#3b7cb8",
            text_color="#ffffff",
            width=380,
            height=45,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        )
        self.btn_capture.pack(pady=8)

        # Upload Button
        self.btn_upload = ctk.CTkButton(
            left_frame,
            text="📁 Upload Image File",
            command=lambda: self.upload_register(user_type, new_user_id),
            fg_color="#3d2a1f",
            hover_color="#4d3a2f",
            border_width=2,
            border_color="#b8733b",
            text_color="#ffffff",
            width=380,
            height=45,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        )
        self.btn_upload.pack(pady=8)

        # Final Register Button
        self.btn_final_register = ctk.CTkButton(
            left_frame,
            text="✓ Complete Registration",
            command=lambda: self.complete_registration(user_type, new_user_id),
            fg_color="#1a3d1a",
            hover_color="#2a4d2a",
            border_width=2,
            border_color="#22c55e",
            text_color="#ffffff",
            width=380,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 14, "bold")
        )
        self.btn_final_register.pack(pady=10)
        self.btn_final_register.configure(state="disabled")

        # Back Button
        ctk.CTkButton(
            left_frame,
            text="← Back",
            command=self.show_register_choice,
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            border_width=2,
            border_color="#555555",
            text_color="#ffffff",
            width=200,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 12)
        ).pack(pady=10)

        # RIGHT SIDE - Camera
        right_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        right_frame.pack(side="right", fill="both", expand=True)

        self.camera_label = ctk.CTkLabel(
            right_frame,
            text="📷 Camera Preview\n\nClick 'Capture from Webcam'\nto start camera",
            width=640,
            height=480,
            fg_color="#1a1a1a",
            corner_radius=15,
            font=("Segoe UI", 14),
            text_color="#888888"
        )
        self.camera_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            right_frame,
            text="Ready to capture",
            font=("Segoe UI", 16, "bold"),
            text_color="#ffcc00"
        )
        self.status_label.pack(pady=10)

    def start_camera_capture(self, user_type, user_id):
        """Start camera for capturing"""
        self.camera_label.configure(text="Starting camera...", fg_color="#1a1a1a")
        self.start_camera()
        self.btn_capture.configure(
            text="📸 Capture Image", 
            command=lambda: self.capture_image(user_type, user_id)
        )

    def capture_image(self, user_type, user_id):
        """Capture image from camera"""
        if self.current_frame is None:
            msg.showerror("Error", "No camera frame available")
            return
            
        enc, status, _ = self.face.process(self.current_frame)
        
        if status != "Face OK":
            msg.showerror("Error", f"Cannot capture: {status}")
            return

        self.captured_encoding = enc
        self.captured_frame = self.current_frame.copy()
        self.image_captured = True
        
        # Stop camera
        self.stop_camera()
        
        # Show captured image
        rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ctk.CTkImage(img, size=(640, 480))
        self.camera_label.configure(image=imgtk, text="")
        self.camera_label.image = imgtk
        
        # Update status
        self.status_label.configure(text="✓ Image Captured Successfully!", text_color="#22c55e")
        
        # Update buttons
        self.btn_capture.configure(
            fg_color="#1a3d1a", 
            border_color="#22c55e", 
            text="✓ Image Captured",
            state="disabled"
        )
        self.btn_final_register.configure(state="normal")

    def upload_register(self, user_type, user_id):
        """Upload image for registration"""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            msg.showerror("Error", "Could not read image file")
            return
            
        enc, status, _ = self.face.process(img)
        
        if status != "Face OK":
            msg.showerror("Error", f"Cannot use this image: {status}")
            return

        self.captured_encoding = enc
        self.captured_frame = img
        self.image_captured = True
        
        # Show uploaded image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        imgtk = ctk.CTkImage(pil_img, size=(640, 480))
        self.camera_label.configure(image=imgtk, text="")
        self.camera_label.image = imgtk
        
        # Update status
        self.status_label.configure(text="✓ Image Uploaded Successfully!", text_color="#22c55e")
        
        # Update buttons
        self.btn_upload.configure(
            fg_color="#1a3d1a", 
            border_color="#22c55e", 
            text="✓ Image Uploaded"
        )
        self.btn_final_register.configure(state="normal")

    def complete_registration(self, user_type, user_id):
        """Complete the registration process"""
        # Validate image
        if not self.image_captured:
            msg.showerror("Error", "Please capture or upload an image first")
            return

        # Validate form
        errors = self.validate_registration_form(user_type)
        if errors:
            msg.showerror("Validation Error", "\n".join(errors))
            return

        # Get form values
        name = self.entry_name.get().strip()
        email = self.entry_email.get().strip()
        age = self.entry_age.get().strip()
        gender = self.gender_var.get()
        phone = self.entry_phone.get().strip()
        dept = self.entry_dept.get().strip()
        pin = self.entry_pin.get().strip() if user_type == "admin" else None

        # Register user
        db.register_user(
            user_id,
            name,
            email,
            age,
            gender,
            phone,
            dept,
            json.dumps(self.captured_encoding.tolist()),
            user_type,
            pin
        )

        # Save image
        user_dir = os.path.join("images/gallery", user_id)
        os.makedirs(user_dir, exist_ok=True)
        cv2.imwrite(os.path.join(user_dir, "register_img.jpg"), self.captured_frame)

        msg.showinfo("Success", f"✓ Registration Successful!\n\nUser ID: {user_id}\n\nYou can now login using your face.")
        self.build_home()

    # ================= LOGIN =================
    def show_login(self):
        """Show login screen"""
        self.clear_screen()
        self.set_background()
        self.image_captured = False

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=80)
        header.pack(fill="x", padx=30, pady=20)

        ctk.CTkLabel(
            header,
            text="IDENTITY FETCH .",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff",
            anchor="w"
        ).pack(side="left")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")
        
        ctk.CTkButton(
            btn_frame,
            text="Identify",
            command=self.show_login,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Register",
            command=self.show_register_choice,
            fg_color="#ffffff",
            text_color="#000000",
            hover_color="#e0e0e0",
            width=120,
            height=40,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)

        # Main card
        card = ctk.CTkFrame(
            self,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=25
        )
        card.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            card,
            text="Identity Verification",
            font=("Segoe UI", 32, "bold"),
            text_color="#ffffff"
        ).pack(pady=(30, 10))

        # Separator
        separator = ctk.CTkFrame(card, height=2, fg_color="#444444")
        separator.pack(fill="x", padx=100, pady=10)

        self.camera_label = ctk.CTkLabel(
            card,
            text="Starting Camera...",
            width=640,
            height=480,
            fg_color="#1a1a1a",
            corner_radius=15
        )
        self.camera_label.pack(padx=30, pady=20)

        self.status_label = ctk.CTkLabel(
            card,
            text="Initializing",
            font=("Segoe UI", 16, "bold"),
            text_color="#ffcc00"
        )
        self.status_label.pack(pady=10)

        self.btn_capture = ctk.CTkButton(
            card,
            text="Verify Face",
            state="disabled",
            command=self.capture_login,
            fg_color="#3b7cb8",
            hover_color="#2d5f8d",
            text_color="#ffffff",
            width=350,
            height=55,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        )
        self.btn_capture.pack(pady=20)

        ctk.CTkButton(
            card,
            text="Back",
            command=self.build_home,
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            border_width=2,
            border_color="#555555",
            text_color="#ffffff",
            width=250,
            height=45,
            corner_radius=50,
            font=("Segoe UI", 14)
        ).pack(pady=(0, 30))
        
        self.start_camera()

    def capture_login(self):
        """Capture and verify face for login"""
        if self.current_frame is None:
            msg.showerror("Error", "No camera frame available")
            return
            
        enc, status, _ = self.face.process(self.current_frame)
        
        if status != "Face OK":
            msg.showerror("Error", f"Cannot verify: {status}")
            return

        df = db.get_all_users()
        if df.empty:
            msg.showerror("Denied", "No users registered")
            return
            
        known = [np.array(json.loads(e)) for e in df["face_encoding"]]
        matches = face_recognition.compare_faces(known, enc, tolerance=0.45)

        if True not in matches:
            attempt_key = "unknown_face"
            if attempt_key not in self.login_attempts:
                self.login_attempts[attempt_key] = {"count": 0, "images": []}
            
            self.login_attempts[attempt_key]["count"] += 1
            self.login_attempts[attempt_key]["images"].append(self.current_frame.copy())
            
            if self.login_attempts[attempt_key]["count"] >= 3:
                today = datetime.now().strftime("%Y-%m-%d")
                breach_dir = os.path.join("images/breach_logs", today)
                os.makedirs(breach_dir, exist_ok=True)
                
                for idx, img in enumerate(self.login_attempts[attempt_key]["images"]):
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"unidentified_{timestamp}_{idx+1}.jpg"
                    cv2.imwrite(os.path.join(breach_dir, filename), img)
                
                self.login_attempts[attempt_key] = {"count": 0, "images": []}
                msg.showerror("Denied", "Face not recognized\nMaximum attempts reached. Logged to breach records.")
            else:
                msg.showerror("Denied", f"Face not recognized\nAttempt {self.login_attempts[attempt_key]['count']}/3")
            return

        self.current_user = df.iloc[matches.index(True)].to_dict()
        
        user_dir = os.path.join("images/gallery", self.current_user["user_id"])
        os.makedirs(user_dir, exist_ok=True)
        login_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(user_dir, f"login_{login_timestamp}.jpg"), self.current_frame)
        
        db.update_login_timestamp(self.current_user["user_id"])
        
        if self.current_user["user_type"] == "admin":
            self.show_pin_dialog()
        else:
            self.current_user_type = "general_user"
            self.show_dashboard()

    def show_pin_dialog(self):
        """Show PIN entry dialog for admin"""
        self.stop_camera()
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("Admin PIN")
        dialog.geometry("500x300")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color="#0a0a0a")
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.winfo_x() + 550, self.winfo_y() + 300))

        frame = ctk.CTkFrame(
            dialog,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="Enter Admin PIN",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=30)

        pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="4-digit PIN",
            width=300,
            height=50,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            show="*",
            font=("Segoe UI", 14),
            justify="center"
        )
        pin_entry.pack(pady=20)
        pin_entry.focus()

        def verify_pin():
            entered_pin = pin_entry.get()
            stored_pin = self.current_user.get("admin_pin")
            if str(entered_pin) == str(stored_pin):
                self.current_user_type = "admin"
                dialog.destroy()
                self.show_dashboard()
            else:
                msg.showerror("Error", "Incorrect PIN")

        ctk.CTkButton(
            frame,
            text="Verify",
            command=verify_pin,
            fg_color="#3b7cb8",
            hover_color="#2d5f8d",
            text_color="#ffffff",
            width=250,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(pady=20)
        
        # Bind Enter key
        pin_entry.bind("<Return>", lambda e: verify_pin())

    # ================= DASHBOARD =================
    def show_dashboard(self):
        """Show dashboard after login"""
        self.clear_screen()
        self.set_background()
        
        user = self.current_user
        is_admin = self.current_user_type == "admin"

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True)

        # SIDEBAR
        sidebar = ctk.CTkFrame(
            main,
            width=280,
            fg_color="#1a1a1a",
            corner_radius=0
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # User info in sidebar
        ctk.CTkLabel(
            sidebar,
            text=f"Welcome,",
            font=("Segoe UI", 14),
            text_color="#888888"
        ).pack(pady=(30, 0))
        
        ctk.CTkLabel(
            sidebar,
            text=user.get("name", "User"),
            font=("Segoe UI", 20, "bold"),
            text_color="#ffffff"
        ).pack(pady=(5, 5))
        
        ctk.CTkLabel(
            sidebar,
            text=f"{'Administrator' if is_admin else 'General User'}",
            font=("Segoe UI", 12),
            text_color="#3b7cb8" if not is_admin else "#6b2d8c"
        ).pack(pady=(0, 30))

        # Separator
        ctk.CTkFrame(sidebar, height=2, fg_color="#333333").pack(fill="x", padx=20, pady=10)

        menu_btn_style = {
            "fg_color": "#2a2a2a",
            "hover_color": "#3a3a3a",
            "border_width": 2,
            "border_color": "#3b7cb8",
            "text_color": "#ffffff",
            "width": 240,
            "height": 50,
            "corner_radius": 50,
            "font": ("Segoe UI", 14, "bold"),
            "anchor": "center"
        }

        ctk.CTkButton(
            sidebar,
            text="👤 My Profile",
            command=lambda: self.show_profile_view(user),
            **menu_btn_style
        ).pack(pady=12, padx=20)

        if is_admin:
            admin_btn_style = menu_btn_style.copy()
            admin_btn_style["border_color"] = "#6b2d8c"
            
            ctk.CTkButton(
                sidebar,
                text="👑 Admin List",
                command=self.show_admin_details,
                **admin_btn_style
            ).pack(pady=12, padx=20)
            
            ctk.CTkButton(
                sidebar,
                text="👥 User List",
                command=self.show_user_details,
                **admin_btn_style
            ).pack(pady=12, padx=20)
            
            ctk.CTkButton(
                sidebar,
                text="⚠️ Breach Logs",
                command=self.show_breach_logs,
                **admin_btn_style
            ).pack(pady=12, padx=20)

        ctk.CTkButton(
            sidebar,
            text="🚪 Logout",
            command=self.logout,
            fg_color="#3d1a1a",
            hover_color="#4d2a2a",
            border_width=2,
            border_color="#d32f2f",
            text_color="#ffffff",
            width=240,
            height=55,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(side="bottom", pady=40, padx=20)

        # Content Area
        self.content_area = ctk.CTkFrame(main, fg_color="transparent")
        self.content_area.pack(side="right", fill="both", expand=True, padx=30, pady=30)

        self.show_profile_view(user)

    def show_profile_view(self, user):
        """Show user profile"""
        for w in self.content_area.winfo_children():
            w.destroy()

        profile_frame = ctk.CTkScrollableFrame(
            self.content_area,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        profile_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(
            profile_frame,
            text="My Profile",
            font=("Segoe UI", 36, "bold"),
            text_color="#ffffff"
        ).pack(pady=30)

        # Circular profile image
        img_paths = db.get_user_images(user["user_id"])
        register_img = [p for p in img_paths if "register" in p]
        if register_img:
            try:
                img = Image.open(register_img[0])
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                
                mask = Image.new('L', (200, 200), 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, 200, 200), fill=255)
                
                output = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
                output.paste(img, (0, 0))
                output.putalpha(mask)
                
                photo = ctk.CTkImage(output, size=(200, 200))
                img_label = ctk.CTkLabel(profile_frame, image=photo, text="")
                img_label.image = photo
                img_label.pack(pady=20)
            except Exception as e:
                print(f"Profile image error: {e}")

        form_frame = ctk.CTkFrame(profile_frame, fg_color="transparent")
        form_frame.pack(pady=20, padx=50)

        fields = [
            ("User ID", user.get("user_id", ""), False),
            ("Name", user.get("name", ""), True),
            ("Email", user.get("email", ""), True),
            ("Age", user.get("age", ""), True),
            ("Gender", user.get("gender", ""), False),
            ("Phone", user.get("phone", ""), True),
            ("Department", user.get("dept", ""), True),
            ("User Type", user.get("user_type", "").replace("_", " ").title(), False),
            ("Last Login", user.get("last_login", "Never"), False),
            ("Last Logout", user.get("last_logout", "Never"), False),
        ]

        entries = {}
        for field, value, editable in fields:
            field_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
            field_frame.pack(pady=8, fill="x")
            
            ctk.CTkLabel(
                field_frame,
                text=field,
                font=("Segoe UI", 14, "bold"),
                text_color="#cccccc",
                width=140,
                anchor="w"
            ).pack(side="left", padx=10)
            
            entry = ctk.CTkEntry(
                field_frame,
                width=350,
                height=42,
                corner_radius=50,
                fg_color="#2a2a2a",
                border_width=2,
                border_color="#333333" if editable else "#222222",
                text_color="#ffffff" if editable else "#888888",
                font=("Segoe UI", 13)
            )
            entry.insert(0, str(value) if value else "")
            if not editable:
                entry.configure(state="disabled")
            entry.pack(side="left", padx=10)
            entries[field] = entry
            
            if editable:
                def save_field(f=field, e=entry):
                    new_val = e.get()
                    field_map = {
                        "Name": "name", 
                        "Email": "email",
                        "Age": "age", 
                        "Phone": "phone", 
                        "Department": "dept"
                    }
                    
                    # Validation
                    if f == "Email" and not self.validate_email(new_val):
                        msg.showerror("Error", "Invalid email format")
                        return
                    if f == "Phone" and not self.validate_phone(new_val):
                        msg.showerror("Error", "Phone must be 10 digits")
                        return
                    if f == "Name" and not self.validate_name(new_val):
                        msg.showerror("Error", "Name must have first and last name")
                        return
                    if f == "Age" and not self.validate_age(new_val):
                        msg.showerror("Error", "Age must be 1-120")
                        return
                    
                    db.update_user_field(user["user_id"], field_map[f], new_val)
                    self.current_user[field_map[f]] = new_val
                    msg.showinfo("Success", f"✓ {f} updated successfully!")
                
                ctk.CTkButton(
                    field_frame,
                    text="Save",
                    command=save_field,
                    fg_color="#1f2d3d",
                    hover_color="#2f3d4d",
                    border_width=2,
                    border_color="#3b7cb8",
                    text_color="#ffffff",
                    width=80,
                    height=38,
                    corner_radius=50,
                    font=("Segoe UI", 12, "bold")
                ).pack(side="left", padx=10)

        # Change PIN button for admin
        if user.get("user_type") == "admin":
            ctk.CTkButton(
                form_frame,
                text="🔐 Change Admin PIN",
                command=lambda: self.change_admin_pin(user["user_id"]),
                fg_color="#2d1f3d",
                hover_color="#3d2f4d",
                border_width=2,
                border_color="#6b2d8c",
                text_color="#ffffff",
                width=300,
                height=50,
                corner_radius=50,
                font=("Segoe UI", 14, "bold")
            ).pack(pady=30)

        # Delete account button
        def delete_account():
            if msg.askyesno("Confirm", "Are you sure you want to delete your account?\nThis action cannot be undone."):
                db.delete_user(user["user_id"])
                msg.showinfo("Success", "✓ Account deleted successfully!")
                self.current_user = None
                self.current_user_type = None
                self.build_home()

        ctk.CTkButton(
            profile_frame,
            text="🗑️ Delete Account",
            command=delete_account,
            fg_color="#3d1a1a",
            hover_color="#4d2a2a",
            border_width=2,
            border_color="#d32f2f",
            text_color="#ffffff",
            width=300,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 14, "bold")
        ).pack(pady=30)

    def change_admin_pin(self, user_id):
        """Dialog to change admin PIN"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Change Admin PIN")
        dialog.geometry("500x450")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color="#0a0a0a")
        dialog.geometry("+%d+%d" % (self.winfo_x() + 550, self.winfo_y() + 225))

        frame = ctk.CTkFrame(
            dialog,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="Change Admin PIN",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=30)

        old_pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Current PIN",
            width=350,
            height=50,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            show="*",
            font=("Segoe UI", 14)
        )
        old_pin_entry.pack(pady=10)

        new_pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="New PIN (4 digits)",
            width=350,
            height=50,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            show="*",
            font=("Segoe UI", 14)
        )
        new_pin_entry.pack(pady=10)

        confirm_pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Confirm New PIN",
            width=350,
            height=50,
            corner_radius=50,
            fg_color="#2a2a2a",
            border_width=2,
            border_color="#333333",
            show="*",
            font=("Segoe UI", 14)
        )
        confirm_pin_entry.pack(pady=10)

        def update_pin():
            old_pin = old_pin_entry.get()
            new_pin = new_pin_entry.get()
            confirm_pin = confirm_pin_entry.get()
            
            if str(old_pin) != str(self.current_user.get("admin_pin")):
                msg.showerror("Error", "Current PIN is incorrect")
                return
            
            if len(new_pin) != 4 or not new_pin.isdigit():
                msg.showerror("Error", "New PIN must be exactly 4 digits")
                return
            
            if new_pin != confirm_pin:
                msg.showerror("Error", "New PINs do not match")
                return
            
            db.update_admin_pin(user_id, new_pin)
            self.current_user["admin_pin"] = new_pin
            msg.showinfo("Success", "✓ Admin PIN updated successfully!")
            dialog.destroy()

        ctk.CTkButton(
            frame,
            text="Update PIN",
            command=update_pin,
            fg_color="#2d1f3d",
            hover_color="#3d2f4d",
            border_width=2,
            border_color="#6b2d8c",
            text_color="#ffffff",
            width=250,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(pady=30)

    def show_admin_details(self):
        """Show all admins (READ-ONLY)"""
        for w in self.content_area.winfo_children():
            w.destroy()

        admin_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        admin_frame.pack(fill="both", expand=True)

        # Header
        header_frame = ctk.CTkFrame(admin_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        ctk.CTkLabel(
            header_frame,
            text="👑 Admin List (Read-Only)",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff"
        ).pack(side="left")

        admins = db.get_users_by_type("admin")
        
        if admins.empty:
            ctk.CTkLabel(
                admin_frame,
                text="No admins found",
                font=("Segoe UI", 18),
                text_color="#cccccc"
            ).pack(pady=50)
            return

        # Table Header
        table_frame = ctk.CTkScrollableFrame(
            admin_frame,
            fg_color="transparent"
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Column headers
        headers = ["User ID", "Name", "Email", "Age", "Gender", "Phone", "Department"]
        header_row = ctk.CTkFrame(table_frame, fg_color="#2a2a2a", corner_radius=10)
        header_row.pack(fill="x", pady=5)
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_row,
                text=header,
                font=("Segoe UI", 12, "bold"),
                text_color="#ffffff",
                width=130
            ).grid(row=0, column=i, padx=5, pady=10)

        # Data rows
        for idx, admin in admins.iterrows():
            row_frame = ctk.CTkFrame(
                table_frame, 
                fg_color="#1f1f1f" if idx % 2 == 0 else "#252525",
                corner_radius=10
            )
            row_frame.pack(fill="x", pady=2)
            
            values = [
                admin.get("user_id", ""),
                admin.get("name", ""),
                admin.get("email", ""),
                admin.get("age", ""),
                admin.get("gender", ""),
                admin.get("phone", ""),
                admin.get("dept", "")
            ]
            
            for i, value in enumerate(values):
                ctk.CTkLabel(
                    row_frame,
                    text=str(value)[:15] + "..." if len(str(value)) > 15 else str(value),
                    font=("Segoe UI", 11),
                    text_color="#cccccc",
                    width=130
                ).grid(row=0, column=i, padx=5, pady=8)

    def show_user_details(self):
        """Show all users (EDITABLE with Delete)"""
        for w in self.content_area.winfo_children():
            w.destroy()

        user_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        user_frame.pack(fill="both", expand=True)

        # Header
        header_frame = ctk.CTkFrame(user_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        ctk.CTkLabel(
            header_frame,
            text="👥 User List (Editable)",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff"
        ).pack(side="left")
        
        # Refresh button
        ctk.CTkButton(
            header_frame,
            text="🔄 Refresh",
            command=self.show_user_details,
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            width=100,
            height=35,
            corner_radius=50
        ).pack(side="right")

        users = db.get_users_by_type("general_user")
        
        if users.empty:
            ctk.CTkLabel(
                user_frame,
                text="No users found",
                font=("Segoe UI", 18),
                text_color="#cccccc"
            ).pack(pady=50)
            return

        # Table
        table_frame = ctk.CTkScrollableFrame(
            user_frame,
            fg_color="transparent"
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Column headers
        headers = ["User ID", "Name", "Email", "Age", "Gender", "Phone", "Dept", "Actions"]
        header_row = ctk.CTkFrame(table_frame, fg_color="#2a2a2a", corner_radius=10)
        header_row.pack(fill="x", pady=5)
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_row,
                text=header,
                font=("Segoe UI", 11, "bold"),
                text_color="#ffffff",
                width=110
            ).grid(row=0, column=i, padx=3, pady=10)

        # Data rows
        for idx, user in users.iterrows():
            row_frame = ctk.CTkFrame(
                table_frame, 
                fg_color="#1f1f1f" if idx % 2 == 0 else "#252525",
                corner_radius=10
            )
            row_frame.pack(fill="x", pady=2)
            
            user_id = user.get("user_id", "")
            
            values = [
                user_id,
                user.get("name", ""),
                user.get("email", ""),
                user.get("age", ""),
                user.get("gender", ""),
                user.get("phone", ""),
                user.get("dept", "")
            ]
            
            for i, value in enumerate(values):
                ctk.CTkLabel(
                    row_frame,
                    text=str(value)[:12] + ".." if len(str(value)) > 12 else str(value),
                    font=("Segoe UI", 10),
                    text_color="#cccccc",
                    width=110
                ).grid(row=0, column=i, padx=3, pady=8)
            
            # Action buttons
            action_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            action_frame.grid(row=0, column=len(values), padx=3, pady=5)
            
            # Edit button
            ctk.CTkButton(
                action_frame,
                text="✏️",
                command=lambda u=user.to_dict(): self.edit_user_dialog(u),
                fg_color="#1f2d3d",
                hover_color="#2f3d4d",
                width=35,
                height=30,
                corner_radius=5
            ).pack(side="left", padx=2)
            
            # Delete button
            ctk.CTkButton(
                action_frame,
                text="🗑️",
                command=lambda uid=user_id: self.delete_user_confirm(uid),
                fg_color="#3d1a1a",
                hover_color="#4d2a2a",
                width=35,
                height=30,
                corner_radius=5
            ).pack(side="left", padx=2)

    def edit_user_dialog(self, user):
        """Dialog to edit user details"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit User")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color="#0a0a0a")
        dialog.geometry("+%d+%d" % (self.winfo_x() + 550, self.winfo_y() + 150))

        frame = ctk.CTkScrollableFrame(
            dialog,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="Edit User Details",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=20)

        # User ID (read-only)
        ctk.CTkLabel(frame, text=f"User ID: {user['user_id']}", text_color="#888888").pack(pady=5)

        # Name
        name_entry = ctk.CTkEntry(frame, placeholder_text="Full Name", width=350, height=45, corner_radius=50, fg_color="#2a2a2a")
        name_entry.insert(0, user.get("name", ""))
        name_entry.pack(pady=8)

        # Email
        email_entry = ctk.CTkEntry(frame, placeholder_text="Email", width=350, height=45, corner_radius=50, fg_color="#2a2a2a")
        email_entry.insert(0, user.get("email", ""))
        email_entry.pack(pady=8)

        # Age
        age_entry = ctk.CTkEntry(frame, placeholder_text="Age", width=350, height=45, corner_radius=50, fg_color="#2a2a2a")
        age_entry.insert(0, str(user.get("age", "")))
        age_entry.pack(pady=8)

        # Phone
        phone_entry = ctk.CTkEntry(frame, placeholder_text="Phone", width=350, height=45, corner_radius=50, fg_color="#2a2a2a")
        phone_entry.insert(0, user.get("phone", ""))
        phone_entry.pack(pady=8)

        # Department
        dept_entry = ctk.CTkEntry(frame, placeholder_text="Department", width=350, height=45, corner_radius=50, fg_color="#2a2a2a")
        dept_entry.insert(0, user.get("dept", ""))
        dept_entry.pack(pady=8)

        def save_changes():
            # Validation
            if not self.validate_name(name_entry.get()):
                msg.showerror("Error", "Invalid name format")
                return
            if not self.validate_email(email_entry.get()):
                msg.showerror("Error", "Invalid email format")
                return
            if not self.validate_age(age_entry.get()):
                msg.showerror("Error", "Invalid age")
                return
            if not self.validate_phone(phone_entry.get()):
                msg.showerror("Error", "Invalid phone number")
                return
            
            db.update_user_details(
                user["user_id"],
                name_entry.get(),
                email_entry.get(),
                age_entry.get(),
                phone_entry.get(),
                dept_entry.get()
            )
            msg.showinfo("Success", "✓ User updated successfully!")
            dialog.destroy()
            self.show_user_details()

        ctk.CTkButton(
            frame,
            text="Save Changes",
            command=save_changes,
            fg_color="#1a3d1a",
            hover_color="#2a4d2a",
            border_width=2,
            border_color="#22c55e",
            width=250,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 14, "bold")
        ).pack(pady=20)

    def delete_user_confirm(self, user_id):
        """Confirm and delete user"""
        if msg.askyesno("Confirm Delete", f"Are you sure you want to delete user {user_id}?\nThis action cannot be undone."):
            db.delete_user(user_id)
            msg.showinfo("Success", "✓ User deleted successfully!")
            self.show_user_details()

    def show_breach_logs(self):
        """Show breach logs"""
        for w in self.content_area.winfo_children():
            w.destroy()

        breach_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        breach_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(
            breach_frame,
            text="⚠️ Breach Logs",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff"
        ).pack(pady=30)

        breach_base = "images/breach_logs"
        
        if not os.path.exists(breach_base) or not os.listdir(breach_base):
            ctk.CTkLabel(
                breach_frame,
                text="No breach logs found",
                font=("Segoe UI", 18),
                text_color="#cccccc"
            ).pack(pady=50)
            return

        # Scrollable frame for logs
        logs_frame = ctk.CTkScrollableFrame(
            breach_frame,
            fg_color="transparent"
        )
        logs_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # List all date folders
        date_folders = sorted(os.listdir(breach_base), reverse=True)
        
        for date_folder in date_folders:
            folder_path = os.path.join(breach_base, date_folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Date header
            date_header = ctk.CTkFrame(logs_frame, fg_color="#2a2a2a", corner_radius=10)
            date_header.pack(fill="x", pady=10)
            
            ctk.CTkLabel(
                date_header,
                text=f"📅 {date_folder}",
                font=("Segoe UI", 16, "bold"),
                text_color="#ffffff"
            ).pack(side="left", padx=20, pady=10)
            
            # Count images
            images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            ctk.CTkLabel(
                date_header,
                text=f"{len(images)} attempts",
                font=("Segoe UI", 12),
                text_color="#ff6b6b"
            ).pack(side="right", padx=20, pady=10)
            
            # Show images in grid
            img_frame = ctk.CTkFrame(logs_frame, fg_color="transparent")
            img_frame.pack(fill="x", pady=5)
            
            for i, img_file in enumerate(images[:6]):  # Show max 6 per date
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path)
                    img = img.resize((150, 120), Image.Resampling.LANCZOS)
                    photo = ctk.CTkImage(img, size=(150, 120))
                    
                    img_label = ctk.CTkLabel(
                        img_frame, 
                        image=photo, 
                        text="",
                        fg_color="#1f1f1f",
                        corner_radius=10
                    )
                    img_label.image = photo
                    img_label.grid(row=i//3, column=i%3, padx=10, pady=5)
                except Exception as e:
                    print(f"Error loading breach image: {e}")

    def logout(self):
        """Logout user"""
        if msg.askyesno("Logout", "Are you sure you want to logout?"):
            if self.current_user:
                db.update_logout_timestamp(self.current_user["user_id"])
            msg.showinfo("Success", "✓ Logged out successfully!")
            self.current_user = None
            self.current_user_type = None
            self.build_home()


if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()