def generate_filename(alg,scaler, pca, outliers):
    base = [alg]
    base.append("_scaler" if scaler else "_no-scaler")
    base.append("_pca" if pca else "_no-pca")
    base.append("_outliers" if outliers else "_no-outliers")
    base.append(".joblib")
    return "".join(base)
