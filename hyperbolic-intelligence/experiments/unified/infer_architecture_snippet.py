# ==============================================================================
# ADAPTIVE ARCHITECTURE INFERENCE
# ==============================================================================
# Cole este snippet no início do notebook para usar arquitetura adaptativa
# em qualquer célula que carrega modelos.

def infer_architecture_from_checkpoint(state_dict: dict) -> dict:
    """
    Infer model architecture from checkpoint state_dict.
    
    Returns dict with:
        - teacher_dim: input dimension
        - hidden_dim: hidden layer dimension  
        - student_dim: output dimension
    
    Example usage:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        arch = infer_architecture_from_checkpoint(state)
        
        model = CGTStudentHardened(
            teacher_dim=arch["teacher_dim"],
            student_dim=arch["student_dim"],
            hidden_dim=arch["hidden_dim"],
        )
        model.load_state_dict(state)
    """
    # Try spectral norm keys first, then regular
    weight_key_0 = None
    weight_key_6 = None
    
    for key in state_dict.keys():
        if 'projector.0.weight' in key and weight_key_0 is None:
            weight_key_0 = key
        if 'projector.6.weight' in key and weight_key_6 is None:
            weight_key_6 = key
    
    if weight_key_0 is None or weight_key_6 is None:
        # Fallback to defaults
        return {
            "teacher_dim": 384,
            "hidden_dim": 256,
            "student_dim": 32,
        }
    
    # projector.0.weight: (hidden_dim, teacher_dim)
    # projector.6.weight: (student_dim, hidden_dim)
    w0 = state_dict[weight_key_0]
    w6 = state_dict[weight_key_6]
    
    return {
        "teacher_dim": w0.shape[1],
        "hidden_dim": w0.shape[0],
        "student_dim": w6.shape[0],
    }


def load_model_adaptive(checkpoint_path, device="cuda"):
    """
    Load model with automatic architecture inference.
    
    Example:
        model = load_model_adaptive("/path/to/checkpoint.pth")
    """
    import torch
    from cgt.models.cgt_hardened import CGTStudentHardened
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    arch = infer_architecture_from_checkpoint(state)
    print(f"[ARCH] Inferred: teacher_dim={arch['teacher_dim']}, "
          f"hidden_dim={arch['hidden_dim']}, student_dim={arch['student_dim']}")
    
    model = CGTStudentHardened(
        teacher_dim=arch["teacher_dim"],
        student_dim=arch["student_dim"],
        hidden_dim=arch["hidden_dim"],
    )
    model.load_state_dict(state)
    model = model.to(device).to(torch.float64)
    model.eval()
    
    return model, arch
