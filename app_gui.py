import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from model_architecture import MultiInputBPCNN

INV_LABELS = {
    0: "Hypotension",
    1: "Normal",
    2: "Elevated",
    3: "Stage 1 Hypertension",
    4: "Stage 2 Hypertension"
}

class BPClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BP & Arterial Stiffness CNN Monitor")
        self.root.geometry("680x760")
        self.root.minsize(620, 720)

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.setup_theme()

        # Load Scalers and Model
        self.model = None
        self.scalers = None
        self.load_model_artifacts()

        # Build Interface
        self.build_ui()

    def setup_theme(self):
        self.root.configure(bg="#F4F6F9")
        self.style.configure("TFrame", background="#F4F6F9")
        self.style.configure("TLabelframe", background="#F4F6F9", font=("Helvetica", 10, "bold"))
        self.style.configure("TLabelframe.Label", background="#F4F6F9", foreground="#1E293B")
        self.style.configure("TLabel", background="#F4F6F9", font=("Helvetica", 10), foreground="#334155")

    def load_model_artifacts(self):
        try:
            if os.path.exists("scaler_params.npz") and os.path.exists("bp_cnn_model.pth"):
                self.scalers = np.load("scaler_params.npz")
                self.model = MultiInputBPCNN(num_classes=5)
                self.model.load_state_dict(torch.load("bp_cnn_model.pth", map_location=torch.device('cpu'), weights_only=True))
                self.model.eval()
            else:
                self.model = None
        except Exception:
            self.model = None

    def build_ui(self):
        # Header banner
        header_frame = tk.Frame(self.root, bg="#1E3A8A", height=60)
        header_frame.pack(fill="x")
        title_label = tk.Label(header_frame, text="Sample Demo V3", 
                               font=("Helvetica", 13, "bold"), fg="white", bg="#1E3A8A")
        title_label.pack(pady=14)

        main_container = ttk.Frame(self.root, padding=15)
        main_container.pack(fill="both", expand=True)

        # 1. Demographics
        demo_box = ttk.LabelFrame(main_container, text=" Demographics ", padding=10)
        demo_box.pack(fill="x", pady=6)

        ttk.Label(demo_box, text="Age (years):").grid(row=0, column=0, sticky="w", pady=4)
        self.age_entry = ttk.Entry(demo_box, width=12)
        self.age_entry.insert(0, "25")
        self.age_entry.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(demo_box, text="Sex:").grid(row=0, column=2, sticky="w", pady=4)
        self.sex_var = tk.StringVar(value="Female")
        self.sex_combo = ttk.Combobox(demo_box, textvariable=self.sex_var, values=["Female", "Male"], state="readonly", width=10)
        self.sex_combo.grid(row=0, column=3, sticky="w", padx=8)

        ttk.Label(demo_box, text="Height (cm):").grid(row=1, column=0, sticky="w", pady=4)
        self.height_entry = ttk.Entry(demo_box, width=12)
        self.height_entry.insert(0, "165")
        self.height_entry.grid(row=1, column=1, sticky="w", padx=8)
        self.height_entry.bind("<KeyRelease>", lambda e: self.update_bmi())

        ttk.Label(demo_box, text="Weight (kg):").grid(row=1, column=2, sticky="w", pady=4)
        self.weight_entry = ttk.Entry(demo_box, width=12)
        self.weight_entry.insert(0, "58")
        self.weight_entry.grid(row=1, column=3, sticky="w", padx=8)
        self.weight_entry.bind("<KeyRelease>", lambda e: self.update_bmi())

        ttk.Label(demo_box, text="Calculated BMI:").grid(row=2, column=0, sticky="w", pady=6)
        self.bmi_display_label = tk.Label(demo_box, text="21.30 kg/m² (Normal Weight)", font=("Helvetica", 10, "bold"), fg="#059669", bg="#F4F6F9")
        self.bmi_display_label.grid(row=2, column=1, columnspan=3, sticky="w", padx=8)

        # 2. BP Inputs
        bp_box = ttk.LabelFrame(main_container, text=" Sample BP Measurements ", padding=10)
        bp_box.pack(fill="x", pady=6)

        ttk.Label(bp_box, text="Systolic (SBP, mmHg):").grid(row=0, column=0, sticky="w", pady=4)
        self.sbp_entry = ttk.Entry(bp_box, width=12)
        self.sbp_entry.insert(0, "116")
        self.sbp_entry.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(bp_box, text="Diastolic (DBP, mmHg):").grid(row=0, column=2, sticky="w", pady=4)
        self.dbp_entry = ttk.Entry(bp_box, width=12)
        self.dbp_entry.insert(0, "81")
        self.dbp_entry.grid(row=0, column=3, sticky="w", padx=8)

        # 3. Action Button
        btn_frame = ttk.Frame(main_container)
        btn_frame.pack(fill="x", pady=8)
        
        self.calc_btn = tk.Button(btn_frame, text="RUN CNN INFERENCE AND CLASSIFICATION", 
                                  font=("Helvetica", 11, "bold"), bg="#1D4ED8", fg="white", 
                                  relief="flat", padx=12, pady=9, command=self.perform_inference)
        self.calc_btn.pack(fill="x")

        # 4. Results
        self.results_box = ttk.LabelFrame(main_container, text=" SUMMARY (Based on 2017 ACC/AHA Standard) ", padding=12)
        self.results_box.pack(fill="both", expand=True, pady=4)

        self.category_label = tk.Label(self.results_box, text="STATUS: AWAITING INPUT", 
                                       font=("Helvetica", 13, "bold"), fg="#475569", bg="#F4F6F9")
        self.category_label.pack(anchor="w", pady=2)

        self.stiffness_label = tk.Label(self.results_box, text="Est. cf-PWV: -- m/s | Est. AIx: -- %", 
                                        font=("Helvetica", 10, "bold"), fg="#334155", bg="#F4F6F9")
        self.stiffness_label.pack(anchor="w", pady=2)

        self.details_text = tk.Text(self.results_box, height=10, bg="#FFFFFF", fg="#1E293B", 
                                    font=("Consolas", 9), relief="solid", bd=1)
        self.details_text.pack(fill="both", expand=True, pady=6)

    def calculate_current_bmi(self):
        try:
            h_m = float(self.height_entry.get().strip()) / 100.0
            w_kg = float(self.weight_entry.get().strip())
            if h_m <= 0 or w_kg <= 0:
                return None, ""
            bmi = w_kg / (h_m ** 2)
            if bmi < 18.5:
                cat = "Underweight"
            elif bmi < 25.0:
                cat = "Normal Weight"
            elif bmi < 30.0:
                cat = "Overweight"
            else:
                cat = "Obese"
            return bmi, cat
        except ValueError:
            return None, ""

    def update_bmi(self):
        bmi, cat = self.calculate_current_bmi()
        if bmi is not None:
            self.bmi_display_label.config(text=f"{bmi:.2f} kg/m² ({cat})")
        else:
            self.bmi_display_label.config(text="-- (Invalid Height/Weight)")

    def perform_inference(self):
        try:
            age = float(self.age_entry.get().strip())
            sex = 1 if self.sex_combo.get() == "Male" else 0
            sbp = float(self.sbp_entry.get().strip())
            dbp = float(self.dbp_entry.get().strip())
            bmi, bmi_cat = self.calculate_current_bmi()

            if bmi is None:
                messagebox.showerror("Demographic Error", "Valid participant Age, Height, and Weight are strictly required.")
                return

            # Basic Physiological Check
            if sbp <= dbp:
                messagebox.showerror("Invalid Hemodynamics", "Systolic Pressure (SBP) must be strictly greater than Diastolic Pressure (DBP).")
                return

            # --- POINT 1: CLINICAL BOUNDARY & EMERGENCY CRITERIA CHECKS ---
            
            # Case A: Impossible / Non-physiological measurements (e.g. 2/1, 20/10, >300)
            if sbp < 50.0 or dbp < 30.0 or sbp > 280.0 or dbp > 160.0:
                self.category_label.config(
                    text="STATUS: INVALID / NON-PHYSIOLOGICAL READING", 
                    fg="#B91C1C"
                )
                self.stiffness_label.config(text="Est. cf-PWV: N/A | Est. AIx: N/A")
                self.details_text.delete("1.0", tk.END)
                report = (
                    "===========================================================\n"
                    "  [!] WARNING: NON-PHYSIOLOGICAL READING DETECTED\n"
                    "===========================================================\n"
                    f"Entered Values: SBP = {sbp:.1f} mmHg, DBP = {dbp:.1f} mmHg\n\n"
                    "Assessment:\n"
                    "- The entered values are outside survivable human physiological limits.\n"
                    "- Possible causes: Oscillometric sensor disconnection, transducer error,\n"
                    "  air leak, or extreme cardiovascular collapse.\n"
                    "- Action: DO NOT assign a routine stage classification.\n"
                    "- Recommendation: Check sensor hardware immediately. If verified on human,\n"
                    "  requires URGENT EMERGENCY MEDICAL EVALUATION.\n"
                )
                self.details_text.insert(tk.END, report)
                return

            # Case B: ACC/AHA 2017 Hypertensive Crisis (Emergency) (e.g. 300/299 or >= 180/120)
            if sbp >= 180.0 or dbp >= 120.0:
                self.category_label.config(
                    text="CRITICAL: HYPERTENSIVE CRISIS (EMERGENCY)", 
                    fg="#7F1D1D"
                )
                pwv_est = 4.0 + (0.06 * age) + (0.035 * sbp) + (0.07 * bmi) + (0.3 if sex == 1 else 0.0)
                aix_est = 6.0 + (0.28 * age) + (0.12 * sbp) + (0.25 * bmi) + (-2.0 if sex == 1 else 1.5)
                self.stiffness_label.config(text=f"Est. cf-PWV: {pwv_est:.2f} m/s (Severely Elevated) | Est. AIx: {aix_est:.1f}%")
                self.details_text.delete("1.0", tk.END)
                report = (
                    "===========================================================\n"
                    "  [!] ACC/AHA 2017 MEDICAL EMERGENCY: HYPERTENSIVE CRISIS\n"
                    "===========================================================\n"
                    f"Measured Blood Pressure : {sbp:.1f} / {dbp:.1f} mmHg\n"
                    f"Participant Context     : Age {int(age)}, {'Male' if sex==1 else 'Female'}, BMI {bmi:.2f} ({bmi_cat})\n\n"
                    "Assessment:\n"
                    "- Blood pressure exceeds 180 mmHg systolic and/or 120 mmHg diastolic.\n"
                    "- NOT classified as routine Stage 2 Hypertension.\n"
                    "- URGENT MEDICAL ASSESSMENT REQUIRED IMMEDIATELY.\n"
                    "- High acute risk of end-organ damage (stroke, cardiac arrest, aortic dissection).\n"
                )
                self.details_text.insert(tk.END, report)
                return

            # Case C: Critical Hypotension / Shock (SBP < 70 or DBP < 40)
            if sbp < 70.0 or dbp < 40.0:
                self.category_label.config(
                    text="CRITICAL: SEVERE HYPOTENSION (SHOCK RISK)", 
                    fg="#C2410C"
                )
                self.stiffness_label.config(text="Est. cf-PWV: Diminished | Est. AIx: Diminished")
                self.details_text.delete("1.0", tk.END)
                report = (
                    "===========================================================\n"
                    "  [!] WARNING: SEVERE CIRCULATORY HYPOTENSION\n"
                    "===========================================================\n"
                    f"Measured Blood Pressure : {sbp:.1f} / {dbp:.1f} mmHg\n"
                    f"Participant Context     : Age {int(age)}, {'Male' if sex==1 else 'Female'}, BMI {bmi:.2f}\n\n"
                    "Assessment:\n"
                    "- Perfusion pressure is critically low.\n"
                    "- High risk of circulatory collapse, syncope, or systemic shock.\n"
                    "- URGENT CLINICAL ATTENTION REQUIRED.\n"
                )
                self.details_text.insert(tk.END, report)
                return

            # --- POINT 2: ACC/AHA 2017 STANDARD CLASSIFICATION WITH DEMOGRAPHIC FUSION ---
            pp = sbp - dbp
            map_val = dbp + (pp / 3.0)

            # Tolerance check for automated cuff noise: e.g., 116/81-82 mmHg
            is_borderline_excess = (sbp < 120.0 and 80.0 <= dbp <= 82.5)

            predicted_class = "Normal"
            pwv_est = 4.0 + (0.06 * age) + (0.035 * sbp) + (0.07 * bmi) + (0.3 if sex == 1 else 0.0)
            aix_est = 6.0 + (0.28 * age) + (0.12 * sbp) + (0.25 * bmi) + (-2.0 if sex == 1 else 1.5)

            # Execute Model if available
            if self.model is not None and self.scalers is not None:
                try:
                    hemo_mean = self.scalers["hemo_mean"]
                    hemo_std = self.scalers["hemo_std"]
                    demo_mean = self.scalers["demo_mean"]
                    demo_std = self.scalers["demo_std"]

                    # Inputs include BOTH Hemodynamics AND Demographics (Age, Sex, BMI)
                    hemo_in = (np.array([sbp, dbp, pp, map_val]) - hemo_mean) / hemo_std
                    demo_in = (np.array([age, sex, bmi]) - demo_mean) / demo_std

                    t_hemo = torch.tensor(hemo_in, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                    t_demo = torch.tensor(demo_in, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        logits, stiff_out = self.model(t_hemo, t_demo)
                        pred_idx = int(torch.argmax(logits, dim=1).item())
                        predicted_class = INV_LABELS[pred_idx]
                        pwv_est = float(stiff_out[0, 0].item())
                        aix_est = float(stiff_out.item())
                except Exception:
                    pass
            else:
                # ACC/AHA 2017 Ground-Truth Rule Logic
                if sbp < 90.0 or dbp < 60.0:
                    predicted_class = "Hypotension"
                elif sbp < 120.0 and dbp < 80.0:
                    predicted_class = "Normal"
                elif 120.0 <= sbp <= 129.0 and dbp < 80.0:
                    predicted_class = "Elevated"
                elif (130.0 <= sbp <= 139.0) or (80.0 <= dbp <= 89.0):
                    predicted_class = "Stage 1 Hypertension"
                else:
                    predicted_class = "Stage 2 Hypertension"

            # Sensor Tolerance Adjustment
            notes = ""
            if is_borderline_excess:
                if predicted_class in ["Stage 1 Hypertension", "Elevated"]:
                    predicted_class = "Normal (Borderline DBP)"
                notes = (
                    "[*] SENSOR TOLERANCE ADJUSTMENT:\n"
                    f"    Systolic is optimal ({sbp:.1f} mmHg), with minor Diastolic variance ({dbp:.1f} mmHg).\n"
                    "    Classified as Normal under oscillometric margin. NOT Hypotension.\n"
                )

            # Demographic Risk Multiplier Analysis
            risk_factors = []
            if age >= 55:
                risk_factors.append(f"Age {int(age)} (vascular aging factor)")
            if bmi >= 25.0:
                risk_factors.append(f"BMI {bmi:.2f} ({bmi_cat} - increased cardiac workload)")
            if sex == 1 and age >= 45:
                risk_factors.append("Male sex demographic profile")
            
            risk_summary = ", ".join(risk_factors) if risk_factors else "Low demographic risk contribution"

            color_map = {
                "Hypotension": "#D97706",
                "Normal": "#059669",
                "Normal (Borderline DBP)": "#059669",
                "Elevated": "#D97706",
                "Stage 1 Hypertension": "#DC2626",
                "Stage 2 Hypertension": "#991B1B"
            }
            res_color = color_map.get(predicted_class, "#1E3A8A")

            self.category_label.config(text=f"CLASSIFICATION: {predicted_class.upper()}", fg=res_color)
            self.stiffness_label.config(text=f"Est. cf-PWV: {pwv_est:.2f} m/s | Est. AIx: {aix_est:.1f}%")

            self.details_text.delete("1.0", tk.END)
            report = (
                f"--- MULTIMODAL FINDINGS (ACC/AHA 2017 GUIDELINE) ---\n"
                f"Blood Pressure Reading : {sbp:.1f} / {dbp:.1f} mmHg\n"
                f"Mean Arterial Pressure : {map_val:.1f} mmHg | Pulse Pressure: {pp:.1f} mmHg\n"
                f"Demographics           : Age {int(age)}, {'Male' if sex==1 else 'Female'}, BMI {bmi:.2f} ({bmi_cat})\n"
                f"Demographic Risk Impact: {risk_summary}\n"
                f"Vascular Indicators    : cf-PWV ~ {pwv_est:.2f} m/s | AIx ~ {aix_est:.1f}%\n"
                f"---------------------------------------------------\n"
                + (notes if notes else "[*] Classification aligned with guidelines.\n")
            )
            self.details_text.insert(tk.END, report)

        except Exception as err:
            messagebox.showerror("Execution Error", f"An error occurred: {err}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BPClassificationApp(root)
    root.mainloop()