import numpy as np

def generate_natural_language_explanation(predicted_mask, cam_heatmap, image_name="image"):
    """
    Analyzes the predicted mask and the Grad-CAM continuous heatmap to generate
    a structured, sectioned clinical XAI report.
    
    Args:
        predicted_mask (np.ndarray): Binary 2D array (H, W). [0=background, 1=lesion]
        cam_heatmap (np.ndarray): Continuous 2D CAM heatmap (H, W), values in [0, 1].
        image_name (str): The source image filename for the report header.

    Returns:
        dict: A dictionary with 'full_report' (str) and 'narrative_only' (str) keys.
    """
    
    total_pixels = predicted_mask.size
    lesion_pixels = int(np.sum(predicted_mask == 1))
    lesion_area_percentage = (lesion_pixels / total_pixels) * 100

    # --------------------------------------------------------------------------
    # Section 1: Segmentation Findings
    # --------------------------------------------------------------------------
    if lesion_pixels == 0:
        seg_section = (
            "The model did not identify any substantive lesion within this image. "
            "The predicted segmentation mask is empty, suggesting this input may not "
            "contain a dermoscopic lesion, or the model confidence was below threshold."
        )
        confidence_label = "No lesion detected"
    else:
        y_indices, x_indices = np.where(predicted_mask == 1)
        mean_y = np.mean(y_indices) / predicted_mask.shape[0]
        mean_x = np.mean(x_indices) / predicted_mask.shape[1]

        v = "central"
        if mean_y < 0.33: v = "upper"
        elif mean_y > 0.66: v = "lower"

        h = "central"
        if mean_x < 0.33: h = "left"
        elif mean_x > 0.66: h = "right"

        location = "central" if (v == "central" and h == "central") else f"{v}-{h}"

        size_label = "small"
        if lesion_area_percentage > 30: size_label = "large"
        elif lesion_area_percentage > 10: size_label = "moderately sized"

        seg_section = (
            f"The model identified a {size_label} lesion covering {lesion_area_percentage:.1f}% "
            f"of the image area in the {location} region of the input dermoscopy image."
        )

    # --------------------------------------------------------------------------
    # Section 2: Model Attention (Grad-CAM)
    # --------------------------------------------------------------------------
    mean_cam = float(np.mean(cam_heatmap))
    peak_cam = float(np.max(cam_heatmap))
    
    # Distribution of high-activation areas
    high_attn_mask = (cam_heatmap > 0.5).astype(int)
    high_attn_pixels = int(np.sum(high_attn_mask))
    
    attn_spread = (high_attn_pixels / total_pixels) * 100
    spread_label = "broadly distributed" if attn_spread > 20 else "focal and concentrated"
    intensity_label = "high-intensity" if peak_cam > 0.75 else "low-intensity"
    
    if lesion_pixels > 0:
        overlap_pixels = int(np.sum((high_attn_mask == 1) & (predicted_mask == 1)))
        overlap_pct = (overlap_pixels / max(high_attn_pixels, 1)) * 100
        
        if overlap_pct > 80:
            overlap_note = "There is strong overlap between activations and the predicted lesion boundary."
        elif overlap_pct > 40:
            overlap_note = "There is moderate overlap — the model focuses partly on the lesion and partly on surrounding skin context."
        else:
            overlap_note = "There is limited overlap — the model may be responding to surrounding skin texture."

        # Determine if activations are central
        if high_attn_pixels > 0:
            attn_y = np.where(high_attn_mask == 1)[0]
            attn_x = np.where(high_attn_mask == 1)[1]
            c_y = np.mean(attn_y) / cam_heatmap.shape[0]
            c_x = np.mean(attn_x) / cam_heatmap.shape[1]
            central = (0.25 < c_y < 0.75) and (0.25 < c_x < 0.75)
            concentration_note = "centrally concentrated" if central else "peripherally distributed"
        else:
            concentration_note = "diffuse"
    else:
        overlap_note = "With no lesion detected, the saliency map shows globally scattered activations."
        concentration_note = "diffuse"
    
    attn_section = (
        f"The Grad-CAM saliency map shows {spread_label}, {intensity_label} activations, "
        f"{concentration_note}. {overlap_note}"
    )

    # --------------------------------------------------------------------------
    # Section 3: Clinical Interpretation
    # --------------------------------------------------------------------------
    if lesion_pixels == 0:
        clinical_section = (
            "No lesion was detected. If clinical suspicion is high, manual dermoscopic "
            "examination is strongly recommended."
        )
    elif overlap_pct > 80:
        clinical_section = (
            "The saliency pattern closely aligns with the predicted segmentation boundary, "
            "suggesting the model is correctly attending to intrinsic lesion features such as "
            "pigmentation, texture irregularities, and structural borders."
        )
    elif overlap_pct > 40:
        clinical_section = (
            "The saliency pattern partially aligns with the lesion region. The model may be "
            "using a combination of lesion-specific and contextual skin features to drive its prediction."
        )
    else:
        clinical_section = (
            "The saliency pattern suggests the model may be relying on contextual image features "
            "rather than intrinsic lesion characteristics. Manual review of the segmentation boundary is advisable."
        )

    # --------------------------------------------------------------------------
    # Section 4: Confidence Assessment
    # --------------------------------------------------------------------------
    if mean_cam > 0.55 and lesion_pixels > 0:
        confidence_label = "High"
        confidence_note = "High average model confidence detected. Segmentation is likely reliable."
    elif mean_cam > 0.35 and lesion_pixels > 0:
        confidence_label = "Moderate"
        confidence_note = "Moderate average confidence detected. Results should be interpreted with care."
    else:
        confidence_label = "Low"
        confidence_note = "Low average confidence detected. The segmentation should be treated with caution and may benefit from a human-in-the-loop review."

    conf_section = f"Overall model confidence is {confidence_label}. {confidence_note}"

    # --------------------------------------------------------------------------
    # Assemble Full Structured Report
    # --------------------------------------------------------------------------
    separator = "-" * 50
    full_report = f"""=== XAI Report - Grad-CAM Analysis ===
Image: {image_name}
{separator}
[Segmentation Findings]
{seg_section}

[Model Attention]
{attn_section}

[Clinical Interpretation]
{clinical_section}

[Confidence Assessment]
{conf_section}

[Disclaimer]
This is an AI-assisted analysis intended to support - not replace - clinical judgement.
All findings should be reviewed by a qualified dermatologist.
"""

    # Narrative-only version (for web display, without the header/footer)
    narrative_only = (
        f"{seg_section}\n\n"
        f"{attn_section}\n\n"
        f"{clinical_section}"
    )

    return {
        'full_report': full_report,
        'narrative_only': narrative_only,
        'confidence': confidence_label
    }
