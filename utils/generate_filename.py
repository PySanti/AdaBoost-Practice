def generate_filename(scaler, smote):
    base = []
    base.append("_scaler" if scaler else "_no-scaler")
    base.append("_smote" if smote else "_no-smote")
    return "".join(base)
