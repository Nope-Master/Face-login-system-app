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
        self.configure(fg_color="#050505")
        
        # Set background
        self.bg_label = None
        self.bg_photo = None
        self.set_background()

        # Initialize system
        db.init_database()
        self.init_folder_structure()
        self.cleanup_old_data()
        
        self.face = FaceSystem()
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
        self.content_area = None

        self.build_home()

    def init_folder_structure(self):
        """Initialize complete folder structure"""
        folders = [
            "images",
            "images/gallery",
            "images/breach_logs",
            "images/unidentified_logs"
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
        
        # Fallback solid color
        self.configure(fg_color="#050505")

    def clear_screen(self):
        """Clear screen but preserve background"""
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
                msg.showerror("Error", "Could not open camera. Please check your camera connection.")
                return False
        self.update_camera()
        return True

    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_camera(self):
        """Update camera feed with face detection overlay"""
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
        
        # Draw rectangles on display frame
        display_frame = frame.copy()
        if face_locs:
            for (top, right, bottom, left) in face_locs:
                # Scale back up since face detection was on smaller frame
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                color = (0, 255, 0) if status == "Face OK" else (0, 165, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)
                
                # Add status text on frame
                cv2.putText(display_frame, status, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert to display format
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ctk.CTkImage(img, size=(640, 480))
        
        # Update camera label if it exists
        if hasattr(self, 'camera_label') and self.camera_label.winfo_exists():
            self.camera_label.configure(image=imgtk, text="")
            self.camera_label.image = imgtk

        # Update status label if it exists
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            if status == "Face OK":
                self.status_label.configure(text=status, text_color="#22c55e")
            elif "No Face" in status:
                self.status_label.configure(text=status, text_color="#ff6b6b")
            else:
                self.status_label.configure(text=status, text_color="#ffcc00")

        # Update capture button state
        if hasattr(self, "btn_capture") and self.btn_capture.winfo_exists():
            if status == "Face OK":
                self.btn_capture.configure(state="normal")
            else:
                self.btn_capture.configure(state="disabled")

        self.after(30, self.update_camera)

    # ================= VALIDATION METHODS =================
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
        """Validate all registration fields and return list of errors"""
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

    # ================= HOME SCREEN =================
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
        """Show registration type selection screen"""
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
            fg_color="rgba(51, 51, 51, 0.85)",
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
        separator = ctk.CTkFrame(card, height=2, fg_color="#ffffff")
        separator.pack(fill="x", padx=100, pady=10)

        ctk.CTkButton(
            card,
            text="Register as Admin",
            command=lambda: self.show_register("admin"),
            fg_color="rgba(107, 45, 140, 0.4)",
            hover_color="rgba(107, 45, 140, 0.6)",
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
            fg_color="rgba(59, 124, 184, 0.4)",
            hover_color="rgba(59, 124, 184, 0.6)",
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
            fg_color="rgba(85, 85, 85, 0.4)",
            hover_color="rgba(85, 85, 85, 0.6)",
            border_width=2,
            border_color="#555555",
            text_color="#ffffff",
            width=250,
            height=45,
            corner_radius=50,
            font=("Segoe UI", 14)
        ).pack(pady=30)

    # ================= REGISTER FORM =================
    def show_register(self, user_type):
        """Show registration form with all fields"""
        self.clear_screen()
        self.set_background()
        self.image_captured = False
        self.captured_encoding = None
        self.captured_frame = None

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
            fg_color="rgba(51, 51, 51, 0.85)",
            border_width=2,
            border_color="#333333",
            corner_radius=25
        )
        card.place(relx=0.5, rely=0.55, anchor="center")

        title = "Admin Registration" if user_type == "admin" else "User Registration"
        ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 32, "bold"),
            text_color="#ffffff"
        ).pack(pady=(30, 10))

        # Separator
        separator = ctk.CTkFrame(card, height=2, fg_color="#ffffff")
        separator.pack(fill="x", padx=100, pady=10)

        # Side-by-side container
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=30, pady=20)

        # LEFT SIDE - Form
        left_frame = ctk.CTkScrollableFrame(
            content_frame, 
            fg_color="transparent",
            width=420,
            height=520
        )
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 15))

        # Generate unique user ID
        new_user_id = f"{'ADM' if user_type == 'admin' else 'USR'}-{uuid.uuid4().hex[:6].upper()}"
        
        # User ID (Auto-generated, read-only)
        ctk.CTkLabel(
            left_frame,
            text="User ID (Auto-generated)",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.entry_userid = ctk.CTkEntry(
            left_frame,
            placeholder_text="User ID",
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            text_color="#888888",
            font=("Segoe UI", 12)
        )
        self.entry_userid.insert(0, new_user_id)
        self.entry_userid.configure(state="disabled")
        self.entry_userid.pack(pady=(0, 8))

        # Full Name
        ctk.CTkLabel(
            left_frame,
            text="Full Name *",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.entry_name = ctk.CTkEntry(
            left_frame,
            placeholder_text="Enter first and last name",
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_name.pack(pady=(0, 8))

        # Email ID
        ctk.CTkLabel(
            left_frame,
            text="Email ID *",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.entry_email = ctk.CTkEntry(
            left_frame,
            placeholder_text="example@email.com",
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_email.pack(pady=(0, 8))

        # Age
        ctk.CTkLabel(
            left_frame,
            text="Age *",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.entry_age = ctk.CTkEntry(
            left_frame,
            placeholder_text="Enter age (1-120)",
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_age.pack(pady=(0, 8))

        # Gender Dropdown
        ctk.CTkLabel(
            left_frame,
            text="Gender *",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.gender_var = ctk.StringVar(value="Select Gender")
        self.gender_dropdown = ctk.CTkOptionMenu(
            left_frame,
            variable=self.gender_var,
            values=["Male", "Female"],
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            button_color="#3b7cb8",
            button_hover_color="#2d5f8d",
            dropdown_fg_color="#2a2a2a",
            dropdown_hover_color="#3a3a3a",
            font=("Segoe UI", 12)
        )
        self.gender_dropdown.pack(pady=(0, 8))

        # Phone Number
        ctk.CTkLabel(
            left_frame,
            text="Phone Number *",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.entry_phone = ctk.CTkEntry(
            left_frame,
            placeholder_text="10 digit phone number",
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_phone.pack(pady=(0, 8))

        # Department
        ctk.CTkLabel(
            left_frame,
            text="Department *",
            font=("Segoe UI", 11),
            text_color="#888888"
        ).pack(anchor="w", padx=10, pady=(5, 0))
        
        self.entry_dept = ctk.CTkEntry(
            left_frame,
            placeholder_text="Enter department",
            width=400,
            height=45,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            font=("Segoe UI", 12)
        )
        self.entry_dept.pack(pady=(0, 8))

        # Admin PIN (only for admin registration)
        if user_type == "admin":
            ctk.CTkLabel(
                left_frame,
                text="Admin PIN *",
                font=("Segoe UI", 11),
                text_color="#888888"
            ).pack(anchor="w", padx=10, pady=(5, 0))
            
            self.entry_pin = ctk.CTkEntry(
                left_frame,
                placeholder_text="4 digit PIN",
                width=400,
                height=45,
                corner_radius=50,
                fg_color="rgba(40, 40, 40, 0.6)",
                border_width=2,
                border_color="#333333",
                show="*",
                font=("Segoe UI", 12)
            )
            self.entry_pin.pack(pady=(0, 8))

        # Required fields note
        ctk.CTkLabel(
            left_frame,
            text="* All fields are required",
            font=("Segoe UI", 10),
            text_color="#ff6b6b"
        ).pack(pady=5)

        # Capture Button
        self.btn_capture = ctk.CTkButton(
            left_frame,
            text="📷 Capture from Webcam",
            command=lambda: self.start_camera_capture(user_type, new_user_id),
            fg_color="rgba(59, 124, 184, 0.4)",
            hover_color="rgba(59, 124, 184, 0.6)",
            border_width=2,
            border_color="#3b7cb8",
            text_color="#ffffff",
            width=400,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        )
        self.btn_capture.pack(pady=10)

        # Upload Button
        self.btn_upload = ctk.CTkButton(
            left_frame,
            text="📁 Upload Image File",
            command=lambda: self.upload_register(user_type, new_user_id),
            fg_color="rgba(184, 115, 59, 0.4)",
            hover_color="rgba(184, 115, 59, 0.6)",
            border_width=2,
            border_color="#b8733b",
            text_color="#ffffff",
            width=400,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 13, "bold")
        )
        self.btn_upload.pack(pady=10)

        # Final Register Button
        self.btn_final_register = ctk.CTkButton(
            left_frame,
            text="✓ Complete Registration",
            command=lambda: self.complete_registration(user_type, new_user_id),
            fg_color="#22c55e",
            hover_color="#16a34a",
            text_color="#ffffff",
            width=400,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 14, "bold"),
            state="disabled"
        )
        self.btn_final_register.pack(pady=10)

        # Back Button
        ctk.CTkButton(
            left_frame,
            text="← Back",
            command=self.show_register_choice,
            fg_color="rgba(85, 85, 85, 0.4)",
            hover_color="rgba(85, 85, 85, 0.6)",
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
            fg_color="rgba(20, 20, 20, 0.7)",
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
        """Start camera for capturing face"""
        self.camera_label.configure(text="Starting camera...", fg_color="rgba(20, 20, 20, 0.7)")
        if self.start_camera():
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
            msg.showerror("Error", f"Cannot capture: {status}\n\nPlease ensure:\n• Good lighting\n• Face clearly visible\n• Only one person in frame\n• Hold still")
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
            fg_color="#22c55e", 
            border_color="#16a34a", 
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
            msg.showerror("Error", f"Cannot use this image: {status}\n\nPlease use an image with:\n• Clear face visible\n• Good lighting\n• Single person only")
            return

        self.captured_encoding = enc
        self.captured_frame = img
        self.image_captured = True
        
        # Stop camera if running
        self.stop_camera()
        
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
            fg_color="#22c55e", 
            border_color="#16a34a", 
            text="✓ Image Uploaded"
        )
        self.btn_final_register.configure(state="normal")

    def complete_registration(self, user_type, user_id):
        """Complete the registration process with validation"""
        # Validate image
        if not self.image_captured:
            msg.showerror("Error", "Please capture or upload an image first")
            return

        # Validate form
        errors = self.validate_registration_form(user_type)
        if errors:
            error_message = "Please fix the following errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            msg.showerror("Validation Error", error_message)
            return

        # Get form values
        name = self.entry_name.get().strip()
        email = self.entry_email.get().strip()
        age = self.entry_age.get().strip()
        gender = self.gender_var.get()
        phone = self.entry_phone.get().strip()
        dept = self.entry_dept.get().strip()
        pin = self.entry_pin.get().strip() if user_type == "admin" else None

        # Register user in database
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

        msg.showinfo("Success", f"✓ Registration Successful!\n\nUser ID: {user_id}\nName: {name}\nType: {user_type.replace('_', ' ').title()}\n\nYou can now login using face recognition.")
        self.build_home()
        
    # ================= LOGIN =================
    def show_login(self):
        """Show login screen with camera"""
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
            fg_color="rgba(51, 51, 51, 0.85)",
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
        separator = ctk.CTkFrame(card, height=2, fg_color="#ffffff")
        separator.pack(fill="x", padx=100, pady=10)

        self.camera_label = ctk.CTkLabel(
            card,
            text="Starting Camera...",
            width=640,
            height=480,
            fg_color="rgba(20, 20, 20, 0.7)",
            corner_radius=15
        )
        self.camera_label.pack(padx=30, pady=20)

        self.status_label = ctk.CTkLabel(
            card,
            text="Initializing...",
            font=("Segoe UI", 16, "bold"),
            text_color="#ffcc00"
        )
        self.status_label.pack(pady=10)

        self.btn_capture = ctk.CTkButton(
            card,
            text="🔍 Verify Face",
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
            fg_color="rgba(85, 85, 85, 0.4)",
            hover_color="rgba(85, 85, 85, 0.6)",
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
            msg.showerror("Denied", "No users registered in the system.\n\nPlease register first.")
            return
            
        known = [np.array(json.loads(e)) for e in df["face_encoding"]]
        matches = face_recognition.compare_faces(known, enc, tolerance=0.45)

        if True not in matches:
            # Handle failed login attempt
            attempt_key = "unknown_face"
            if attempt_key not in self.login_attempts:
                self.login_attempts[attempt_key] = {"count": 0, "images": []}
            
            self.login_attempts[attempt_key]["count"] += 1
            self.login_attempts[attempt_key]["images"].append(self.current_frame.copy())
            
            if self.login_attempts[attempt_key]["count"] >= 3:
                # Save breach images
                today = datetime.now().strftime("%Y-%m-%d")
                breach_dir = os.path.join("images/breach_logs", today)
                os.makedirs(breach_dir, exist_ok=True)
                
                for idx, img in enumerate(self.login_attempts[attempt_key]["images"]):
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"unidentified_{timestamp}_{idx+1}.jpg"
                    cv2.imwrite(os.path.join(breach_dir, filename), img)
                
                self.login_attempts[attempt_key] = {"count": 0, "images": []}
                msg.showerror("Access Denied", "⚠️ Face not recognized!\n\nMaximum attempts reached.\nThis incident has been logged to breach records.")
            else:
                remaining = 3 - self.login_attempts[attempt_key]['count']
                msg.showerror("Access Denied", f"Face not recognized\n\nAttempts remaining: {remaining}")
            return

        # Successful match
        self.current_user = df.iloc[matches.index(True)].to_dict()
        
        # Save login image
        user_dir = os.path.join("images/gallery", self.current_user["user_id"])
        os.makedirs(user_dir, exist_ok=True)
        login_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(user_dir, f"login_{login_timestamp}.jpg"), self.current_frame)
        
        # Update login timestamp
        db.update_login_timestamp(self.current_user["user_id"])
        
        # Check if admin needs PIN
        if self.current_user["user_type"] == "admin":
            self.show_pin_dialog()
        else:
            self.current_user_type = "general_user"
            self.stop_camera()
            self.show_dashboard()

    def show_pin_dialog(self):
        """Show PIN entry dialog for admin verification"""
        self.stop_camera()
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("Admin PIN Verification")
        dialog.geometry("500x320")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color="#050505")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.winfo_x() + 550, self.winfo_y() + 290))

        frame = ctk.CTkFrame(
            dialog,
            fg_color="rgba(51, 51, 51, 0.9)",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="🔐 Enter Admin PIN",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=30)

        pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="••••",
            width=300,
            height=50,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            show="•",
            font=("Segoe UI", 18),
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
                msg.showerror("Error", "Incorrect PIN. Please try again.")
                pin_entry.delete(0, 'end')

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
        """Show dashboard after successful login"""
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
            fg_color="rgba(30, 30, 30, 0.95)",
            corner_radius=0
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # User welcome section
        ctk.CTkLabel(
            sidebar,
            text="Welcome,",
            font=("Segoe UI", 14),
            text_color="#888888"
        ).pack(pady=(40, 0))
        
        ctk.CTkLabel(
            sidebar,
            text=user.get("name", "User"),
            font=("Segoe UI", 20, "bold"),
            text_color="#ffffff"
        ).pack(pady=(5, 5))
        
        role_text = "👑 Administrator" if is_admin else "👤 General User"
        role_color = "#6b2d8c" if is_admin else "#3b7cb8"
        ctk.CTkLabel(
            sidebar,
            text=role_text,
            font=("Segoe UI", 12),
            text_color=role_color
        ).pack(pady=(0, 30))

        # Separator
        ctk.CTkFrame(sidebar, height=2, fg_color="#333333").pack(fill="x", padx=20, pady=10)

        # Menu button style
        menu_btn_style = {
            "fg_color": "rgba(59, 124, 184, 0.3)",
            "hover_color": "rgba(59, 124, 184, 0.5)",
            "border_width": 2,
            "border_color": "rgba(59, 124, 184, 0.6)",
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
            admin_btn_style.update({
                "fg_color": "rgba(107, 45, 140, 0.3)",
                "hover_color": "rgba(107, 45, 140, 0.5)",
                "border_color": "rgba(107, 45, 140, 0.6)"
            })
            
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

        # Logout button at bottom
        ctk.CTkButton(
            sidebar,
            text="🚪 Logout",
            command=self.logout,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            text_color="#ffffff",
            width=240,
            height=55,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(side="bottom", pady=40, padx=20)

        # Content Area
        self.content_area = ctk.CTkFrame(main, fg_color="transparent")
        self.content_area.pack(side="right", fill="both", expand=True, padx=30, pady=30)

        # Show profile by default
        self.show_profile_view(user)

    def show_profile_view(self, user):
        """Show user profile with editable fields"""
        for w in self.content_area.winfo_children():
            w.destroy()

        profile_frame = ctk.CTkScrollableFrame(
            self.content_area,
            fg_color="rgba(51, 51, 51, 0.85)",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        profile_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(
            profile_frame,
            text="👤 My Profile",
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
                
                # Create circular mask
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

        # Define fields with editability
        fields = [
            ("User ID", user.get("user_id", ""), False),
            ("Name", user.get("name", ""), True),
            ("Email", user.get("email", ""), True),
            ("Age", user.get("age", ""), True),
            ("Gender", user.get("gender", ""), False),
            ("Phone", user.get("phone", ""), True),
            ("Department", user.get("dept", ""), True),
            ("User Type", str(user.get("user_type", "")).replace("_", " ").title(), False),
            ("Created At", user.get("created_at", "N/A"), False),
            ("Last Login", user.get("last_login", "Never"), False),
            ("Last Logout", user.get("last_logout", "Never"), False),
        ]

        entries = {}
        for field, value, editable in fields:
            field_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
            field_frame.pack(pady=10, fill="x")
            
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
                height=45,
                corner_radius=50,
                fg_color="rgba(40, 40, 40, 0.6)",
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
                        msg.showerror("Error", "Invalid email format (e.g., user@example.com)")
                        return
                    if f == "Phone" and not self.validate_phone(new_val):
                        msg.showerror("Error", "Phone must be exactly 10 digits")
                        return
                    if f == "Name" and not self.validate_name(new_val):
                        msg.showerror("Error", "Name must have first and last name (letters only)")
                        return
                    if f == "Age" and not self.validate_age(new_val):
                        msg.showerror("Error", "Age must be between 1 and 120")
                        return
                    
                    db.update_user_field(user["user_id"], field_map[f], new_val)
                    self.current_user[field_map[f]] = new_val
                    msg.showinfo("Success", f"✓ {f} updated successfully!")
                
                ctk.CTkButton(
                    field_frame,
                    text="Save",
                    command=save_field,
                    fg_color="rgba(59, 124, 184, 0.4)",
                    hover_color="rgba(59, 124, 184, 0.6)",
                    border_width=2,
                    border_color="#3b7cb8",
                    text_color="#ffffff",
                    width=100,
                    height=40,
                    corner_radius=50,
                    font=("Segoe UI", 12, "bold")
                ).pack(side="left", padx=10)

        # Change PIN button for admin
        if user.get("user_type") == "admin":
            ctk.CTkButton(
                form_frame,
                text="🔐 Change Admin PIN",
                command=lambda: self.change_admin_pin(user["user_id"]),
                fg_color="rgba(107, 45, 140, 0.4)",
                hover_color="rgba(107, 45, 140, 0.6)",
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
            if msg.askyesno("Confirm Delete", "Are you sure you want to delete your account?\n\nThis action cannot be undone.\nAll your data and images will be permanently deleted."):
                db.delete_user(user["user_id"])
                msg.showinfo("Success", "✓ Account deleted successfully!")
                self.current_user = None
                self.current_user_type = None
                self.build_home()

        ctk.CTkButton(
            profile_frame,
            text="🗑️ Delete Account",
            command=delete_account,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
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
        dialog.configure(fg_color="#050505")
        dialog.resizable(False, False)
        dialog.geometry("+%d+%d" % (self.winfo_x() + 550, self.winfo_y() + 225))

        frame = ctk.CTkFrame(
            dialog,
            fg_color="rgba(51, 51, 51, 0.9)",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="🔐 Change Admin PIN",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=30)

        old_pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Current PIN",
            width=350,
            height=50,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            show="•",
            font=("Segoe UI", 14)
        )
        old_pin_entry.pack(pady=10)

        new_pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="New PIN (4 digits)",
            width=350,
            height=50,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            show="•",
            font=("Segoe UI", 14)
        )
        new_pin_entry.pack(pady=10)

        confirm_pin_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Confirm New PIN",
            width=350,
            height=50,
            corner_radius=50,
            fg_color="rgba(40, 40, 40, 0.6)",
            border_width=2,
            border_color="#333333",
            show="•",
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
            fg_color="rgba(107, 45, 140, 0.4)",
            hover_color="rgba(107, 45, 140, 0.6)",
            border_width=2,
            border_color="#6b2d8c",
            text_color="#ffffff",
            width=250,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 16, "bold")
        ).pack(pady=30)

    def show_admin_details(self):
        """Show all admins in a table (READ-ONLY)"""
        for w in self.content_area.winfo_children():
            w.destroy()

        admin_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="rgba(51, 51, 51, 0.85)",
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
                text_color="#888888"
            ).pack(pady=50)
            return

        # Table container
        table_frame = ctk.CTkScrollableFrame(
            admin_frame,
            fg_color="transparent"
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Column headers
        headers = ["User ID", "Name", "Email", "Age", "Gender", "Phone", "Department"]
        header_row = ctk.CTkFrame(table_frame, fg_color="rgba(42, 42, 42, 0.9)", corner_radius=10)
        header_row.pack(fill="x", pady=5)
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_row,
                text=header,
                font=("Segoe UI", 12, "bold"),
                text_color="#ffffff",
                width=130
            ).grid(row=0, column=i, padx=5, pady=12)

        # Data rows
        for idx, admin in admins.iterrows():
            row_color = "rgba(31, 31, 31, 0.9)" if idx % 2 == 0 else "rgba(37, 37, 37, 0.9)"
            row_frame = ctk.CTkFrame(table_frame, fg_color=row_color, corner_radius=10)
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
                display_text = str(value)[:15] + "..." if len(str(value)) > 15 else str(value)
                ctk.CTkLabel(
                    row_frame,
                    text=display_text,
                    font=("Segoe UI", 11),
                    text_color="#cccccc",
                    width=130
                ).grid(row=0, column=i, padx=5, pady=10)

    def show_user_details(self):
        """Show all users in a table (EDITABLE with Delete)"""
        for w in self.content_area.winfo_children():
            w.destroy()

        user_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="rgba(51, 51, 51, 0.85)",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        user_frame.pack(fill="both", expand=True)

        # Header with refresh button
        header_frame = ctk.CTkFrame(user_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        ctk.CTkLabel(
            header_frame,
            text="👥 User List (Editable)",
            font=("Segoe UI", 28, "bold"),
            text_color="#ffffff"
        ).pack(side="left")
        
        ctk.CTkButton(
            header_frame,
            text="🔄 Refresh",
            command=self.show_user_details,
            fg_color="rgba(42, 42, 42, 0.9)",
            hover_color="rgba(58, 58, 58, 0.9)",
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
                text_color="#888888"
            ).pack(pady=50)
            return

        # Table container
        table_frame = ctk.CTkScrollableFrame(
            user_frame,
            fg_color="transparent"
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Column headers
        headers = ["User ID", "Name", "Email", "Age", "Gender", "Phone", "Dept", "Actions"]
        header_row = ctk.CTkFrame(table_frame, fg_color="rgba(42, 42, 42, 0.9)", corner_radius=10)
        header_row.pack(fill="x", pady=5)
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_row,
                text=header,
                font=("Segoe UI", 11, "bold"),
                text_color="#ffffff",
                width=110
            ).grid(row=0, column=i, padx=3, pady=12)

        # Data rows
        for idx, user in users.iterrows():
            row_color = "rgba(31, 31, 31, 0.9)" if idx % 2 == 0 else "rgba(37, 37, 37, 0.9)"
            row_frame = ctk.CTkFrame(table_frame, fg_color=row_color, corner_radius=10)
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
                display_text = str(value)[:12] + ".." if len(str(value)) > 12 else str(value)
                ctk.CTkLabel(
                    row_frame,
                    text=display_text,
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
                fg_color="rgba(59, 124, 184, 0.5)",
                hover_color="rgba(59, 124, 184, 0.7)",
                width=35,
                height=30,
                corner_radius=5
            ).pack(side="left", padx=2)
            
            # Delete button
            ctk.CTkButton(
                action_frame,
                text="🗑️",
                command=lambda uid=user_id: self.delete_user_confirm(uid),
                fg_color="rgba(211, 47, 47, 0.5)",
                hover_color="rgba(211, 47, 47, 0.7)",
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
        dialog.configure(fg_color="#050505")
        dialog.geometry("+%d+%d" % (self.winfo_x() + 550, self.winfo_y() + 150))

        frame = ctk.CTkScrollableFrame(
            dialog,
            fg_color="rgba(51, 51, 51, 0.9)",
            border_width=2,
            border_color="#333333",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="✏️ Edit User Details",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffffff"
        ).pack(pady=20)

        ctk.CTkLabel(
            frame,
            text=f"User ID: {user['user_id']}",
            font=("Segoe UI", 12),
            text_color="#888888"
        ).pack(pady=5)

        # Name
        name_entry = ctk.CTkEntry(frame, placeholder_text="Full Name", width=350, height=45, corner_radius=50, fg_color="rgba(40, 40, 40, 0.6)")
        name_entry.insert(0, user.get("name", ""))
        name_entry.pack(pady=8)

        # Email
        email_entry = ctk.CTkEntry(frame, placeholder_text="Email", width=350, height=45, corner_radius=50, fg_color="rgba(40, 40, 40, 0.6)")
        email_entry.insert(0, user.get("email", ""))
        email_entry.pack(pady=8)

        # Age
        age_entry = ctk.CTkEntry(frame, placeholder_text="Age", width=350, height=45, corner_radius=50, fg_color="rgba(40, 40, 40, 0.6)")
        age_entry.insert(0, str(user.get("age", "")))
        age_entry.pack(pady=8)

        # Phone
        phone_entry = ctk.CTkEntry(frame, placeholder_text="Phone", width=350, height=45, corner_radius=50, fg_color="rgba(40, 40, 40, 0.6)")
        phone_entry.insert(0, user.get("phone", ""))
        phone_entry.pack(pady=8)

        # Department
        dept_entry = ctk.CTkEntry(frame, placeholder_text="Department", width=350, height=45, corner_radius=50, fg_color="rgba(40, 40, 40, 0.6)")
        dept_entry.insert(0, user.get("dept", ""))
        dept_entry.pack(pady=8)

        def save_changes():
            # Validation
            if not self.validate_name(name_entry.get()):
                msg.showerror("Error", "Invalid name format. Must have first and last name.")
                return
            if not self.validate_email(email_entry.get()):
                msg.showerror("Error", "Invalid email format")
                return
            if not self.validate_age(age_entry.get()):
                msg.showerror("Error", "Invalid age. Must be 1-120.")
                return
            if not self.validate_phone(phone_entry.get()):
                msg.showerror("Error", "Invalid phone. Must be 10 digits.")
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
            text="💾 Save Changes",
            command=save_changes,
            fg_color="#22c55e",
            hover_color="#16a34a",
            text_color="#ffffff",
            width=250,
            height=50,
            corner_radius=50,
            font=("Segoe UI", 14, "bold")
        ).pack(pady=20)

    def delete_user_confirm(self, user_id):
        """Confirm and delete user"""
        if msg.askyesno("Confirm Delete", f"Are you sure you want to delete user:\n{user_id}?\n\nThis action cannot be undone."):
            db.delete_user(user_id)
            msg.showinfo("Success", "✓ User deleted successfully!")
            self.show_user_details()

    def show_breach_logs(self):
        """Show breach logs with images"""
        for w in self.content_area.winfo_children():
            w.destroy()

        breach_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="rgba(51, 51, 51, 0.85)",
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
                text="No breach logs found\n\nThis is good! No unauthorized access attempts detected.",
                font=("Segoe UI", 16),
                text_color="#22c55e"
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
            date_header = ctk.CTkFrame(logs_frame, fg_color="rgba(42, 42, 42, 0.9)", corner_radius=10)
            date_header.pack(fill="x", pady=10)
            
            ctk.CTkLabel(
                date_header,
                text=f"📅 {date_folder}",
                font=("Segoe UI", 16, "bold"),
                text_color="#ffffff"
            ).pack(side="left", padx=20, pady=12)
            
            # Count images
            images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            ctk.CTkLabel(
                date_header,
                text=f"{len(images)} unauthorized attempts",
                font=("Segoe UI", 12),
                text_color="#ff6b6b"
            ).pack(side="right", padx=20, pady=12)
            
            # Show images in grid
            img_frame = ctk.CTkFrame(logs_frame, fg_color="transparent")
            img_frame.pack(fill="x", pady=5)
            
            for i, img_file in enumerate(images[:6]):  # Show max 6 per date
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path)
                    img = img.resize((160, 120), Image.Resampling.LANCZOS)
                    photo = ctk.CTkImage(img, size=(160, 120))
                    
                    img_container = ctk.CTkFrame(img_frame, fg_color="rgba(31, 31, 31, 0.9)", corner_radius=10)
                    img_container.grid(row=i//3, column=i%3, padx=10, pady=8)
                    
                    img_label = ctk.CTkLabel(img_container, image=photo, text="")
                    img_label.image = photo
                    img_label.pack(padx=5, pady=5)
                    
                    # Timestamp from filename
                    ctk.CTkLabel(
                        img_container,
                        text=img_file.split("_")[1] if "_" in img_file else "Unknown",
                        font=("Segoe UI", 9),
                        text_color="#888888"
                    ).pack(pady=(0, 5))
                    
                except Exception as e:
                    print(f"Error loading breach image: {e}")
            
            if len(images) > 6:
                ctk.CTkLabel(
                    logs_frame,
                    text=f"... and {len(images) - 6} more images",
                    font=("Segoe UI", 11),
                    text_color="#888888"
                ).pack(pady=5)

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