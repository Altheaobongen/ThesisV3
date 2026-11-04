import numpy as np
import pandas as pd

def generate_acc_aha_dataset(n_samples=1200, random_seed=42):
    np.random.seed(random_seed)
    data = []
    
    classes = [
        "Hypotension",
        "Normal",
        "Elevated",
        "Stage 1 Hypertension",
        "Stage 2 Hypertension"
    ]
    
    samples_per_class = n_samples // len(classes)

    for cls in classes:
        for _ in range(samples_per_class):
            sex = int(np.random.randint(0, 2))  # 0: Female, 1: Male
            
            if cls == "Hypotension":
                age = int(np.random.normal(24, 6))
                age = int(np.clip(age, 18, 60))
                bmi = np.random.normal(18.5, 1.8)
                bmi = float(np.clip(bmi, 15.0, 23.0))
                sbp = np.random.uniform(70.0, 89.5)
                dbp = np.random.uniform(40.0, 59.5)

            elif cls == "Normal":
                age = int(np.random.normal(32, 10))
                age = int(np.clip(age, 18, 65))
                bmi = np.random.normal(22.0, 2.5)
                bmi = float(np.clip(bmi, 18.5, 25.5))
                # Sensor tolerance up to 82 DBP if SBP is well within normal (<120)
                sbp = np.random.uniform(90.0, 119.5)
                dbp = np.random.uniform(60.0, 82.0)

            elif cls == "Elevated":
                # ACC/AHA: SBP 120-129 AND DBP < 80
                age = int(np.random.normal(42, 11))
                age = int(np.clip(age, 25, 75))
                bmi = np.random.normal(26.0, 2.8)
                bmi = float(np.clip(bmi, 22.0, 30.0))
                sbp = np.random.uniform(120.0, 129.5)
                dbp = np.random.uniform(62.0, 79.5)

            elif cls == "Stage 1 Hypertension":
                # ACC/AHA: SBP 130-139 OR DBP 80-89
                age = int(np.random.normal(50, 12))
                age = int(np.clip(age, 28, 80))
                bmi = np.random.normal(28.5, 3.2)
                bmi = float(np.clip(bmi, 23.5, 35.0))
                pattern = np.random.choice(["sbp_only", "dbp_only", "both"])
                if pattern == "sbp_only":
                    sbp = np.random.uniform(130.0, 139.5)
                    dbp = np.random.uniform(70.0, 79.5)
                elif pattern == "dbp_only":
                    sbp = np.random.uniform(110.0, 129.5)
                    dbp = np.random.uniform(80.0, 89.5)
                else:
                    sbp = np.random.uniform(130.0, 139.5)
                    dbp = np.random.uniform(80.0, 89.5)

            elif cls == "Stage 2 Hypertension":
                # ACC/AHA: SBP 140-179 OR DBP 90-119 (>=180 or >=120 is Crisis)
                age = int(np.random.normal(58, 11))
                age = int(np.clip(age, 35, 80))
                bmi = np.random.normal(31.0, 4.0)
                bmi = float(np.clip(bmi, 25.0, 42.0))
                sbp = np.random.uniform(140.0, 179.5)
                dbp = np.random.uniform(90.0, 119.5)

            pp = sbp - dbp
            map_val = dbp + (pp / 3.0)
            
            # Arterial stiffness incorporating Age, Sex, BMI, and Pressure
            pwv = 4.0 + (0.06 * age) + (0.035 * sbp) + (0.07 * bmi) + (0.3 if sex == 1 else 0.0) + np.random.normal(0, 0.35)
            aix = 6.0 + (0.28 * age) + (0.12 * sbp) + (0.25 * bmi) + (-2.0 if sex == 1 else 1.5) + np.random.normal(0, 1.0)

            data.append({
                "age": age,
                "sex": sex,
                "bmi": round(float(bmi), 2),
                "sbp": round(float(sbp), 1),
                "dbp": round(float(dbp), 1),
                "pulse_pressure": round(float(pp), 1),
                "map": round(float(map_val), 1),
                "pwv": round(float(pwv), 2),
                "aix": round(float(aix), 1),
                "target_class": cls
            })

    df = pd.DataFrame(data)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df.to_csv("acc_aha_bp_dataset.csv", index=False)
    print(f"Generated {len(df)} clinically bounded samples in 'acc_aha_bp_dataset.csv'.")

if __name__ == "__main__":
    generate_acc_aha_dataset()