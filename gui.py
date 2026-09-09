"""
gui.py

Desktop interface for the Face Recognition Attendance System, built with
CustomTkinter. Each screen (Dashboard, Students, Register, Attendance,
Records) is its own Frame subclass, all sharing one InsightFace model
instance and the database.py persistence layer.
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from recognition.insightface_adapter import InsightFaceAdapter
from recognition.matcher import match_embedding
import database as db

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

ACCENT = "#2E6FF2"
GREEN = "#1FA96B"
RED = "#E5484D"
ORANGE = "#E08E22"
GREY = "#8A8A8A"


class FRSApp(ctk.CTk):
    """Main application window: owns the shared recognition model,
    the navigation sidebar, and all five screens."""

    def __init__(self):
        super().__init__()
        self.title("Face Attendance System")
        self.geometry("1050x680")
        self.configure(fg_color="#F5F6FA")

        try:
            db.init_db()
            db.migrate_db()
        except Exception as e:
            messagebox.showerror("Database error", f"Could not initialize the database.\n\n{e}")
            self.destroy(); sys.exit(1)

        print("Loading recognition model...")
        try:
            self.adapter = InsightFaceAdapter()
        except Exception as e:
            messagebox.showerror("Recognition model error", f"Could not load the InsightFace model.\n\n{e}")
            self.destroy(); sys.exit(1)
        print("Model loaded.")

        nav = ctk.CTkFrame(self, width=200, fg_color="white", corner_radius=0)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)

        ctk.CTkLabel(nav, text="Face Attendance", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#1a1a1a").pack(pady=(30, 40), padx=20, anchor="w")

        self.nav_buttons = {}
        for label, name in [("Dashboard", "dashboard"), ("Students", "students"),
                             ("Register Student", "register"), ("Take Attendance", "attendance"),
                             ("Attendance Records", "records")]:
            btn = ctk.CTkButton(nav, text=label, anchor="w", fg_color="transparent",
                                 text_color="#444", hover_color="#EEF2FF",
                                 font=ctk.CTkFont(size=13), height=40, corner_radius=8,
                                 command=lambda n=name: self.show_frame(n))
            btn.pack(fill="x", padx=15, pady=3)
            self.nav_buttons[name] = btn

        container = ctk.CTkFrame(self, fg_color="#F5F6FA")
        container.pack(side="right", fill="both", expand=True)

        self.frames = {}
        for F, name in [(DashboardFrame, "dashboard"), (StudentsFrame, "students"),
                         (RegisterFrame, "register"), (AttendanceFrame, "attendance"),
                         (RecordsFrame, "records")]:
            frame = F(container, self)
            self.frames[name] = frame
            frame.place(relwidth=1, relheight=1)

        self.show_frame("dashboard")

    def show_frame(self, name):
        for n, btn in self.nav_buttons.items():
            btn.configure(fg_color="#EEF2FF" if n == name else "transparent",
                          text_color=ACCENT if n == name else "#444")
        frame = self.frames[name]
        if hasattr(frame, "on_show"):
            frame.on_show()
        frame.tkraise()

    def on_close(self):
        for frame in self.frames.values():
            if hasattr(frame, "stop_camera"):
                frame.stop_camera()
        self.destroy()


def stat_card(parent, title, value):
    card = ctk.CTkFrame(parent, fg_color="white", corner_radius=12, width=200, height=100)
    card.pack_propagate(False)
    ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13), text_color=GREY).pack(anchor="w", padx=20, pady=(18, 0))
    value_label = ctk.CTkLabel(card, text=str(value), font=ctk.CTkFont(size=28, weight="bold"), text_color="#1a1a1a")
    value_label.pack(anchor="w", padx=20, pady=(0, 15))
    return card, value_label


def styled_table():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Custom.Treeview", background="white", fieldbackground="white",
                     rowheight=32, borderwidth=0, font=("Helvetica", 11))
    style.configure("Custom.Treeview.Heading", background="#F5F6FA", font=("Helvetica", 11, "bold"), relief="flat")
    style.map("Custom.Treeview", background=[("selected", "#EEF2FF")])
    return style


class DashboardFrame(ctk.CTkFrame):
    """Landing screen showing quick stats: registered students and
    today's attendance count."""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="#F5F6FA")
        ctk.CTkLabel(self, text="Dashboard", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#1a1a1a").pack(pady=(30, 20), padx=30, anchor="w")
        cards_row = ctk.CTkFrame(self, fg_color="transparent")
        cards_row.pack(anchor="w", padx=30)
        c1, self.students_val = stat_card(cards_row, "Students Registered", "0")
        c1.pack(side="left", padx=(0, 15))
        c2, self.today_val = stat_card(cards_row, "Today's Attendance", "0")
        c2.pack(side="left", padx=(0, 15))

    def on_show(self):
        try:
            self.students_val.configure(text=str(db.count_students()))
            self.today_val.configure(text=str(db.count_today_attendance()))
        except Exception as e:
            messagebox.showerror("Database error", str(e))


class StudentsFrame(ctk.CTkFrame):
    """Admin view of all students (active and inactive) supporting
    editing details, toggling active status and deletion."""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="#F5F6FA")
        ctk.CTkLabel(self, text="Students", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#1a1a1a").pack(pady=(30, 20), padx=30, anchor="w")

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(padx=30, anchor="w", pady=(0, 10))
        ctk.CTkButton(btns, text="Edit Student", fg_color=ACCENT, hover_color="#2058C8",
                      height=32, command=self.edit_selected).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btns, text="Toggle Active/Inactive", fg_color=ORANGE, hover_color="#B87118",
                      height=32, command=self.toggle_active).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btns, text="Delete Student", fg_color=RED, hover_color="#C23A3E",
                      height=32, command=self.delete_selected).pack(side="left")

        styled_table()
        columns = ("student_id", "name", "course", "year", "duration", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", style="Custom.Treeview")
        for col in columns:
            self.tree.heading(col, text=col.replace("_", " ").title())
        self.tree.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        self.row_ids = {}

    def on_show(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.row_ids = {}
        try:
            for s in db.list_all_students_including_inactive():
                status = "Active" if s["active"] else "Inactive"
                item = self.tree.insert("", "end", values=(s["student_id"], s["name"], s["course"],
                                                             s["year"], s.get("course_duration") or "-", status))
                self.row_ids[item] = s["id"]
        except Exception as e:
            messagebox.showerror("Database error", f"Could not load students.\n\n{e}")

    def get_selected_id(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No row selected", "Select a student first.")
            return None
        return self.row_ids.get(selection[0])

    def edit_selected(self):
        internal_id = self.get_selected_id()
        if internal_id is None:
            return
        student = db.get_student_by_internal_id(internal_id)

        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit Student")
        dialog.geometry("340x300")
        fields = {}
        for i, (label, key) in enumerate([("Name", "name"), ("Course", "course"),
                                           ("Year", "year"), ("Course Duration (yrs)", "course_duration")]):
            ctk.CTkLabel(dialog, text=label, font=ctk.CTkFont(size=12)).pack(pady=(12, 2), padx=20, anchor="w")
            e = ctk.CTkEntry(dialog, width=280)
            e.pack(padx=20)
            e.insert(0, student.get(key) or "")
            fields[key] = e

        def save():
            try:
                db.update_student(internal_id, fields["name"].get().strip(), fields["course"].get().strip(),
                                   fields["year"].get().strip(), fields["course_duration"].get().strip())
            except Exception as e:
                messagebox.showerror("Database error", str(e))
                return
            dialog.destroy()
            self.on_show()

        ctk.CTkButton(dialog, text="Save", fg_color=ACCENT, command=save).pack(pady=20)

    def toggle_active(self):
        internal_id = self.get_selected_id()
        if internal_id is None:
            return
        student = db.get_student_by_internal_id(internal_id)
        new_state = not bool(student["active"])
        try:
            db.set_student_active(internal_id, new_state)
        except Exception as e:
            messagebox.showerror("Database error", str(e))
            return
        self.on_show()

    def delete_selected(self):
        internal_id = self.get_selected_id()
        if internal_id is None:
            return
        student = db.get_student_by_internal_id(internal_id)
        student_name = student['name']
        student_code = student['student_id']
        msg = 'Permanently delete ' + student_name + ' (' + student_code + ')?\n\n'
        msg += 'This also removes their face samples and attendance history. This cannot be undone.'
        confirm = messagebox.askyesno('Delete student', msg)
        if not confirm:
            return
        try:
            db.delete_student(internal_id)
        except Exception as e:
            messagebox.showerror('Database error', str(e))
            return
        self.on_show()


class RegisterFrame(ctk.CTkFrame):
    """Captures a new student's details and several live face samples,
    saving one embedding per captured sample to the database."""

    SAMPLES_NEEDED = 5

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="#F5F6FA")
        self.app = app
        self.cap = None
        self.internal_id = None
        self.samples_captured = 0
        self.last_faces = []

        ctk.CTkLabel(self, text="Register Student", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#1a1a1a").pack(pady=(30, 20), padx=30, anchor="w")

        card = ctk.CTkFrame(self, fg_color="white", corner_radius=12)
        card.pack(padx=30, pady=(0, 20), fill="x")

        form = ctk.CTkFrame(card, fg_color="transparent")
        form.pack(padx=25, pady=25, anchor="w")
        self.entries = {}
        for i, field in enumerate(["Student ID", "Name", "Course", "Year", "Course Duration (yrs)"]):
            ctk.CTkLabel(form, text=field, font=ctk.CTkFont(size=12), text_color=GREY,
                         width=140, anchor="w").grid(row=i, column=0, sticky="w", pady=6)
            e = ctk.CTkEntry(form, width=260, height=34)
            e.grid(row=i, column=1, pady=6, padx=(10, 0))
            self.entries[field] = e

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(padx=25, pady=(0, 20), anchor="w")
        ctk.CTkButton(btns, text="Start Camera", fg_color=ACCENT, hover_color="#2058C8",
                      height=36, command=self.start_camera).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btns, text="Capture Sample", fg_color=GREEN, hover_color="#178A56",
                      height=36, command=self.capture_sample).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btns, text="Stop Camera", fg_color="transparent", border_width=1,
                      border_color="#ccc", text_color="#444", hover_color="#EEE",
                      height=36, command=self.stop_camera).pack(side="left")

        self.status_label = ctk.CTkLabel(self, text="Camera not started.", text_color=GREY,
                                          font=ctk.CTkFont(size=13))
        self.status_label.pack(pady=(0, 10), padx=30, anchor="w")

        self.video_label = ctk.CTkLabel(self, text="", fg_color="#1a1a1a", corner_radius=12,
                                         width=480, height=340)
        self.video_label.pack(padx=30)

    def start_camera(self):
        student_id = self.entries["Student ID"].get().strip()
        name = self.entries["Name"].get().strip()
        course = self.entries["Course"].get().strip()
        year = self.entries["Year"].get().strip()
        duration = self.entries["Course Duration (yrs)"].get().strip()

        if not student_id or not name:
            messagebox.showerror("Missing information", "Student ID and Name are both required.")
            return

        try:
            existing = db.get_student_by_student_id(student_id)
            if existing:
                self.internal_id = existing["id"]
                if existing["name"].strip().lower() != name.lower():
                    proceed = messagebox.askyesno(
                        "Student ID already exists",
                        f"Student ID '{student_id}' is already registered as '{existing['name']}'.\n\n"
                        f"Add new face samples to that existing record anyway?"
                    )
                    if not proceed:
                        return
            else:
                self.internal_id = db.add_student(student_id, name, course, year, duration)
        except Exception as e:
            messagebox.showerror("Database error", f"Could not register student.\n\n{e}")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera unavailable",
                                  "Could not access the webcam. Check permissions and that no "
                                  "other app is using it.")
            self.cap = None
            return

        self.samples_captured = 0
        self.status_label.configure(text=f"{self.samples_captured}/{self.SAMPLES_NEEDED} samples captured")
        self.update_frame()

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.configure(text="Lost connection to camera.")
            self.stop_camera()
            return

        faces = self.app.adapter.detect_faces(frame)
        self.last_faces = faces

        display = frame.copy()
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (46, 111, 242), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb).resize((480, 340)))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.after(15, self.update_frame)

    def capture_sample(self):
        if self.cap is None:
            messagebox.showerror("No camera", "Start the camera first.")
            return
        if len(self.last_faces) == 0:
            messagebox.showwarning("No face detected", "Please move into the camera frame and try again.")
            return
        if len(self.last_faces) > 1:
            messagebox.showwarning("Multiple faces detected", "Only one person should be in frame.")
            return

        try:
            embedding = self.app.adapter.get_embedding(self.last_faces[0])
            db.save_embedding(self.internal_id, embedding)
        except Exception as e:
            messagebox.showerror("Error saving sample", str(e))
            return

        self.samples_captured += 1
        self.status_label.configure(text=f"{self.samples_captured}/{self.SAMPLES_NEEDED} samples captured")

        if self.samples_captured >= self.SAMPLES_NEEDED:
            messagebox.showinfo("Registration complete", f"{self.samples_captured} sample(s) saved.")
            self.stop_camera()
            for e in self.entries.values():
                e.delete(0, 'end')

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="")


class AttendanceFrame(ctk.CTkFrame):
    """Live camera recognition loop for a chosen class/session, marking
    attendance for recognised active students and preventing duplicates
    within the same session."""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="#F5F6FA")
        self.app = app
        self.cap = None
        self.session_id = None
        self.known_embeddings = {}
        self.marked_this_run = set()

        ctk.CTkLabel(self, text="Take Attendance", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#1a1a1a").pack(pady=(30, 20), padx=30, anchor="w")

        card = ctk.CTkFrame(self, fg_color="white", corner_radius=12)
        card.pack(padx=30, pady=(0, 15), fill="x")

        form = ctk.CTkFrame(card, fg_color="transparent")
        form.pack(padx=25, pady=20, anchor="w")
        ctk.CTkLabel(form, text="Class", text_color=GREY, font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self.class_entry = ctk.CTkEntry(form, width=200, height=34)
        self.class_entry.grid(row=1, column=0, padx=(0, 15))
        ctk.CTkLabel(form, text="Course", text_color=GREY, font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w")
        self.course_entry = ctk.CTkEntry(form, width=200, height=34)
        self.course_entry.grid(row=1, column=1)

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(padx=25, pady=(0, 20), anchor="w")
        ctk.CTkButton(btns, text="Start Attendance", fg_color=ACCENT, hover_color="#2058C8",
                      height=36, command=self.start).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btns, text="Stop", fg_color="transparent", border_width=1, border_color="#ccc",
                      text_color="#444", hover_color="#EEE", height=36,
                      command=self.stop_camera).pack(side="left")

        self.video_label = ctk.CTkLabel(self, text="", fg_color="#1a1a1a", corner_radius=12,
                                         width=480, height=320)
        self.video_label.pack(padx=30)

        table_card = ctk.CTkFrame(self, fg_color="white", corner_radius=12)
        table_card.pack(padx=30, pady=15, fill="x")
        styled_table()
        columns = ("name", "status", "time")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", style="Custom.Treeview", height=5)
        for col in columns:
            self.tree.heading(col, text=col.title())
        self.tree.pack(fill="x", padx=15, pady=15)

    def start(self):
        class_name = self.class_entry.get().strip()
        course = self.course_entry.get().strip()
        if not class_name:
            messagebox.showerror("Missing information", "Class name is required.")
            return

        try:
            existing_class = db.get_class_by_name(class_name)
            class_id = existing_class["id"] if existing_class else db.add_class(class_name, course)
            self.session_id = db.get_or_create_todays_session(class_id)
            self.known_embeddings = db.load_all_embeddings()
        except Exception as e:
            messagebox.showerror("Database error", str(e))
            return

        if not self.known_embeddings:
            messagebox.showwarning("No students registered",
                                    "No active registered students found. Register or reactivate "
                                    "a student first.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera unavailable", "Could not access the webcam.")
            self.cap = None
            return

        for row in self.tree.get_children():
            self.tree.delete(row)
        self.marked_this_run = set()
        self.update_frame()

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera error", "Lost connection to the camera.")
            self.stop_camera()
            return

        faces = self.app.adapter.detect_faces(frame)
        display = frame.copy()

        for face in faces:
            box = face.bbox.astype(int)
            embedding = self.app.adapter.get_embedding(face)
            student_id, score = match_embedding(embedding, self.known_embeddings)

            if student_id:
                try:
                    student = db.get_student_by_student_id(student_id)
                    if student_id in self.marked_this_run:
                        label = f"{student['name']} - Already Marked"
                        color = (0, 165, 255)
                    else:
                        was_marked = db.mark_attendance(self.session_id, student["id"], score)
                        self.marked_this_run.add(student_id)
                        if was_marked:
                            label = f"{student['name']} - Present"
                            color = (46, 169, 31)
                            self.tree.insert("", 0, values=(student["name"], "● Present",
                                                              datetime.now().strftime("%H:%M:%S")))
                        else:
                            label = f"{student['name']} - Already Marked"
                            color = (0, 165, 255)
                except Exception as e:
                    label = "Error"; color = (0, 165, 255)
                    print(f"Error marking attendance: {e}")
            else:
                label = "Unknown - Not Registered"
                color = (77, 72, 229)

            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(box[1] - 15, text_h + 10)
            cv2.rectangle(display, (box[0], label_y - text_h - baseline - 6),
                          (box[0] + text_w + 12, label_y + baseline - 2), color, -1)
            cv2.putText(display, label, (box[0] + 6, label_y - 4), font, font_scale, (255, 255, 255), thickness)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb).resize((480, 320)))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.after(15, self.update_frame)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="")


class RecordsFrame(ctk.CTkFrame):
    """Searchable, sortable table of all attendance records, with
    editing, duplication and deletion support."""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="#F5F6FA")
        self.all_records = []
        self.sort_state = {}

        ctk.CTkLabel(self, text="Attendance Records", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#1a1a1a").pack(pady=(30, 15), padx=30, anchor="w")

        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(padx=30, fill="x", pady=(0, 10))

        self.search_entry = ctk.CTkEntry(top_row, width=280, height=32,
                                          placeholder_text="Search by name, ID, or class...")
        self.search_entry.pack(side="left", padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda e: self.apply_filter())

        ctk.CTkButton(top_row, text="Edit Record", fg_color=ACCENT, hover_color="#2058C8",
                      height=32, command=self.edit_selected).pack(side="left", padx=(0, 10))
        ctk.CTkButton(top_row, text="Duplicate Record", fg_color=GREEN, hover_color="#178A56",
                      height=32, command=self.duplicate_selected).pack(side="left", padx=(0, 10))
        ctk.CTkButton(top_row, text="Delete Record", fg_color=RED, hover_color="#C23A3E",
                      height=32, command=self.delete_selected).pack(side="left")

        styled_table()
        columns = ("date", "name", "student_id", "class_name", "timestamp", "confidence")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", style="Custom.Treeview")
        for col in columns:
            self.tree.heading(col, text=col.replace("_", " ").title(),
                              command=lambda c=col: self.sort_by(c))
        self.tree.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        self.row_ids = {}

    def on_show(self):
        try:
            self.all_records = db.list_all_attendance()
        except Exception as e:
            messagebox.showerror("Database error", str(e))
            return
        self.apply_filter()

    def apply_filter(self):
        query = self.search_entry.get().strip().lower()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.row_ids = {}

        filtered = self.all_records
        if query:
            filtered = [r for r in self.all_records if
                        query in r["name"].lower() or
                        query in r["student_id"].lower() or
                        query in r["class_name"].lower()]

        for r in filtered:
            item = self.tree.insert("", "end", values=(r["date"], r["name"], r["student_id"],
                                                         r["class_name"], r["timestamp"],
                                                         f"{r['confidence']:.3f}"))
            self.row_ids[item] = r["id"]

    def sort_by(self, column):
        ascending = self.sort_state.get(column, True)
        self.all_records.sort(key=lambda r: str(r[column]), reverse=not ascending)
        self.sort_state[column] = not ascending
        self.apply_filter()

    def get_selected_record_id(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No row selected", "Select a record first.")
            return None
        return self.row_ids.get(selection[0])

    def edit_selected(self):
        record_id = self.get_selected_record_id()
        if record_id is None:
            return
        matches = [r for r in self.all_records if r["id"] == record_id]
        current = matches[0]
        parts = current["timestamp"].split(" ")
        current_date, current_time = parts[0], parts[1]

        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit Attendance Record")
        dialog.geometry("340x320")

        fields = {}
        field_defs = [("Class Name", "class", current["class_name"]),
                      ("Date (YYYY-MM-DD)", "date", current_date),
                      ("Time (HH:MM:SS)", "time", current_time)]
        for label, key, default in field_defs:
            ctk.CTkLabel(dialog, text=label, font=ctk.CTkFont(size=12)).pack(pady=(15, 2), padx=20, anchor="w")
            e = ctk.CTkEntry(dialog, width=280)
            e.pack(padx=20)
            e.insert(0, default)
            fields[key] = e

        def save():
            class_name = fields["class"].get().strip()
            date_str = fields["date"].get().strip()
            time_str = fields["time"].get().strip()
            try:
                datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                messagebox.showerror("Invalid format", "Use Date: YYYY-MM-DD and Time: HH:MM:SS")
                return
            try:
                db.update_attendance_full(record_id, class_name, date_str, time_str)
            except Exception as e:
                messagebox.showerror("Database error", str(e))
                return
            dialog.destroy()
            self.on_show()

        ctk.CTkButton(dialog, text="Save", fg_color=ACCENT, command=save).pack(pady=20)

    def duplicate_selected(self):
        record_id = self.get_selected_record_id()
        if record_id is None:
            return
        try:
            db.duplicate_attendance(record_id)
        except Exception as e:
            messagebox.showerror("Database error", str(e))
            return
        self.on_show()

    def delete_selected(self):
        record_id = self.get_selected_record_id()
        if record_id is None:
            return
        confirm = messagebox.askyesno("Delete record", "Delete this attendance record? This cannot be undone.")
        if not confirm:
            return
        try:
            db.delete_attendance(record_id)
        except Exception as e:
            messagebox.showerror("Database error", str(e))
            return
        self.on_show()


if __name__ == "__main__":
    app = FRSApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
